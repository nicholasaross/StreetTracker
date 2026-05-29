"""DVSA MOT history API client for plate -> make/model lookups.

The MOT history API at ``https://history.mot.api.gov.uk`` returns
make + model + year + primary colour + fuel type given a UK
registration number. We use it to auto-label vehicles seen by
StreetTracker without any hand-labelling step, and as the ground-truth
regression test for the in-house make/model classifier (planned in
the make/model design doc).

Authentication is OAuth2 client-credentials (Microsoft Entra). The
client obtains a short-lived Bearer token (TTL ~3600 s) at startup
and refreshes it automatically when it nears expiry. A long-lived
40-char ``X-API-Key`` is also required on every request.

Rate limit (per DVSA docs): 15 RPS sustained, 10-req burst,
500,000 / day. The client throttles to 15 RPS via a per-call sleep,
which is the right tradeoff for batch-mode accumulation -- the data
labelling pass is bottlenecked by the number of plates, not request
latency, so a token bucket is over-engineering.

See ``configs/dvsa.example.json`` for the credential file format.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# DVSA's documented limit. We deliberately stay just below it so a
# brief network blip doesn't push us over.
_RATE_LIMIT_RPS = 15.0
_MIN_INTERVAL_S = 1.0 / _RATE_LIMIT_RPS

# Refresh the OAuth token when this many seconds remain on the cached
# one. Anything > a few HTTP-roundtrip latencies guarantees we never
# send a stale token on a slow network.
_TOKEN_REFRESH_BUFFER_S = 60.0

# Connect + read timeout for each API call. The MOT API typically
# responds in < 500 ms; 10 s is generous and tolerates one TCP retry.
_REQUEST_TIMEOUT_S = 10.0


@dataclasses.dataclass(frozen=True, slots=True)
class OAuthConfig:
    """Microsoft Entra client-credentials parameters."""

    client_id: str
    client_secret: str
    token_url: str
    scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class DvsaConfig:
    """Top-level credentials block, loaded from ``configs/dvsa.json``."""

    x_api_key: str
    oauth: OAuthConfig

    @classmethod
    def from_json_file(cls, path: Path) -> DvsaConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "_comment" in raw:
            raw.pop("_comment")
        try:
            return cls(
                x_api_key=raw["x_api_key"],
                oauth=OAuthConfig(
                    client_id=raw["oauth"]["client_id"],
                    client_secret=raw["oauth"]["client_secret"],
                    token_url=raw["oauth"]["token_url"],
                    scope=raw["oauth"]["scope"],
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"{path}: missing required key {exc.args[0]!r}. "
                f"See configs/dvsa.example.json for the schema."
            ) from exc


@dataclasses.dataclass(frozen=True, slots=True)
class MotLookupResult:
    """Trimmed-down per-plate result -- only the fields the labelling
    pass needs, not the full MOT test history."""

    registration: str
    make: str
    model: str
    year: int | None              # parsed from firstUsedDate
    primary_colour: str | None
    fuel_type: str | None
    engine_size_cc: int | None
    looked_up_at: str             # ISO 8601 UTC timestamp


@dataclasses.dataclass(slots=True)
class _Token:
    access_token: str
    expires_at_mono: float        # time.monotonic() value when token dies


def _parse_year(date_str: str | None) -> int | None:
    """Extract the year from ``YYYY-MM-DD`` strings the MOT API returns.
    Returns None if the string is None / empty / malformed."""
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    """The MOT API stringifies numeric fields (e.g. ``"3800"``).
    Coerce to int; return None on failure."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, AttributeError):
        return None


class DvsaClient:
    """Synchronous DVSA MOT history API client.

    Designed for batch labelling: one client instance per session
    accumulator run. Thread-safety is not provided -- the rate-limit
    sleep would fight a multi-threaded caller and the token cache
    isn't guarded. Use one instance per process.
    """

    def __init__(
        self,
        config: DvsaConfig,
        *,
        session: requests.Session | None = None,
        api_base: str = "https://history.mot.api.gov.uk",
    ) -> None:
        self.config = config
        self.api_base = api_base.rstrip("/")
        # Allow injection for tests; default to a fresh session per client.
        self._session = session if session is not None else requests.Session()
        self._token: _Token | None = None
        self._last_request_mono: float = 0.0

    # ------------------------------------------------------------------
    # Token management.

    def _token_is_fresh(self) -> bool:
        return (
            self._token is not None
            and time.monotonic() < self._token.expires_at_mono - _TOKEN_REFRESH_BUFFER_S
        )

    def _refresh_token(self) -> None:
        oauth = self.config.oauth
        resp = self._session.post(
            oauth.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": oauth.client_id,
                "client_secret": oauth.client_secret,
                "scope": oauth.scope,
            },
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        body = resp.json()
        try:
            access_token = body["access_token"]
            expires_in = float(body["expires_in"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DVSA token endpoint returned malformed body: {body!r}"
            ) from exc
        self._token = _Token(
            access_token=access_token,
            expires_at_mono=time.monotonic() + expires_in,
        )
        logger.info(
            "[dvsa] obtained Bearer token (expires in %.0f s)", expires_in
        )

    def _ensure_token(self) -> str:
        if not self._token_is_fresh():
            self._refresh_token()
        assert self._token is not None
        return self._token.access_token

    # ------------------------------------------------------------------
    # Rate limiting.

    def _throttle(self) -> None:
        """Sleep enough to keep us under 15 RPS."""
        now = time.monotonic()
        delta = now - self._last_request_mono
        if delta < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - delta)
        self._last_request_mono = time.monotonic()

    # ------------------------------------------------------------------
    # Public API.

    def lookup_plate(self, registration: str) -> MotLookupResult | None:
        """Look up a UK registration. Returns ``None`` for 404 (plate
        not on file -- vehicle < 3 years old, exported, or DVSA simply
        has no record). Raises for any other non-2xx response so a
        broken pipeline / wrong credentials surface immediately rather
        than silently producing empty labels.
        """
        vrm = registration.upper().replace(" ", "")
        if not vrm:
            return None
        self._throttle()
        token = self._ensure_token()
        url = f"{self.api_base}/v1/trade/vehicles/registration/{vrm}"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-API-Key": self.config.x_api_key,
            "Accept": "application/json",
        }
        resp = self._session.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)
        if resp.status_code == 404:
            return None
        if resp.status_code == 401:
            # Token may have been revoked mid-run; one retry after a
            # forced refresh covers the common race.
            logger.info("[dvsa] 401 on %s -- refreshing token + retrying", vrm)
            self._token = None
            token = self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = self._session.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)
        if resp.status_code == 429:
            # Backoff briefly and retry once. If we keep hitting 429 the
            # caller's throttle is too aggressive; let the exception
            # bubble on the second hit so the operator notices.
            retry_after = float(resp.headers.get("Retry-After", "2"))
            logger.warning("[dvsa] 429 -- sleeping %.1f s + retrying %s", retry_after, vrm)
            time.sleep(retry_after)
            resp = self._session.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        body = resp.json()
        return MotLookupResult(
            registration=body.get("registration", vrm),
            make=body.get("make", "") or "",
            model=body.get("model", "") or "",
            year=_parse_year(body.get("firstUsedDate") or body.get("registrationDate")),
            primary_colour=body.get("primaryColour"),
            fuel_type=body.get("fuelType"),
            engine_size_cc=_parse_int(body.get("engineSize")),
            looked_up_at=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
        )
