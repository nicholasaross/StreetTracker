"""DVSA MOT API client tests.

The point of these tests is to lock down the auth + retry + 404
semantics without hitting the live DVSA API. Real responses (from the
operator's auth.txt smoke test) are used as fixtures so the parser
stays honest about field shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from streettracker.analysis.dvsa import (
    DvsaClient,
    DvsaConfig,
    MotLookupResult,
    OAuthConfig,
    _parse_int,
    _parse_year,
    is_canonical_uk_plate,
)


@pytest.fixture
def dvsa_config() -> DvsaConfig:
    return DvsaConfig(
        x_api_key="test_api_key_40_chars_aaaaaaaaaaaaaaaaaa",
        oauth=OAuthConfig(
            client_id="00000000-0000-0000-0000-000000000000",
            client_secret="test_secret",
            token_url="https://example.invalid/oauth/token",
            scope="https://tapi.dvsa.gov.uk/.default",
        ),
    )


def _token_response(expires_in: float = 3600.0) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "token_type": "Bearer",
        "expires_in": expires_in,
        "access_token": "test_bearer_token",
    }
    return resp


def _mot_response(body: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = body
    return resp


def _err_response(status: int, *, retry_after: str | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.headers = {"Retry-After": retry_after} if retry_after else {}
    if 400 <= status < 600:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return resp


# Real shape, lifted from the operator's AE13SJX smoke test in DVSA/.
_REAL_PORSCHE_BODY = {
    "registration": "AE13SJX",
    "make": "PORSCHE",
    "model": "911",
    "firstUsedDate": "2013-03-28",
    "fuelType": "Petrol",
    "primaryColour": "Black",
    "registrationDate": "2013-03-28",
    "manufactureDate": "2013-03-28",
    "engineSize": "3800",
    "hasOutstandingRecall": "Unknown",
    "motTests": [],
}


# ----------------------------------------------------------------------
# DvsaConfig.from_json_file


def test_config_from_json_file_round_trip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "dvsa.json"
    cfg_path.write_text(
        json.dumps(
            {
                "x_api_key": "k",
                "oauth": {
                    "client_id": "cid",
                    "client_secret": "sec",
                    "token_url": "https://example.invalid/t",
                    "scope": "https://example.invalid/.default",
                },
            }
        )
    )
    cfg = DvsaConfig.from_json_file(cfg_path)
    assert cfg.x_api_key == "k"
    assert cfg.oauth.client_id == "cid"
    assert cfg.oauth.scope == "https://example.invalid/.default"


def test_config_from_json_file_strips_comment(tmp_path: Path) -> None:
    """The .example file ships with a _comment array; loader must accept it."""
    cfg_path = tmp_path / "dvsa.json"
    cfg_path.write_text(
        json.dumps(
            {
                "_comment": ["explanatory text"],
                "x_api_key": "k",
                "oauth": {
                    "client_id": "cid",
                    "client_secret": "sec",
                    "token_url": "u",
                    "scope": "s",
                },
            }
        )
    )
    cfg = DvsaConfig.from_json_file(cfg_path)
    assert cfg.x_api_key == "k"


def test_config_missing_key_raises_with_path_hint(tmp_path: Path) -> None:
    cfg_path = tmp_path / "dvsa.json"
    cfg_path.write_text(json.dumps({"x_api_key": "k"}))
    with pytest.raises(ValueError, match="oauth"):
        DvsaConfig.from_json_file(cfg_path)


# ----------------------------------------------------------------------
# _parse_year / _parse_int


def test_parse_year_handles_iso_date() -> None:
    assert _parse_year("2013-03-28") == 2013
    assert _parse_year("1999-12-31") == 1999


def test_parse_year_handles_none_and_garbage() -> None:
    assert _parse_year(None) is None
    assert _parse_year("") is None
    assert _parse_year("abc") is None
    assert _parse_year("12") is None  # too short


def test_parse_int_coerces_stringified_numerics() -> None:
    """MOT API returns engineSize as a string -- coerce robustly."""
    assert _parse_int("3800") == 3800
    assert _parse_int(3800) == 3800
    assert _parse_int(" 1998 ") == 1998
    assert _parse_int(None) is None
    assert _parse_int("not a number") is None


# ----------------------------------------------------------------------
# is_canonical_uk_plate


@pytest.mark.parametrize(
    "plate",
    [
        # Current (2001+): LL00LLL
        "AE13SJX", "VK74TPO", "LD22BMG", "BC75ANF",
        # Prefix (1983-2001): L1-3 LLL
        "X123ABC", "A1BCD", "Y99XYZ",
        # Suffix (1963-1983): LLL1-3 L
        "ABC123X", "XYZ1A", "ABC99Z",
    ],
)
def test_is_canonical_uk_plate_accepts_all_three_uk_schemes(plate: str) -> None:
    assert is_canonical_uk_plate(plate) is True


@pytest.mark.parametrize(
    "plate",
    [
        # OCR garbage from the 2026-05-29 re-soak smoke test
        "1157WHT",       # leading digit, post-2001 shape broken
        "1214LZH",
        "123889",
        "141232",
        "142219",
        "1E6971",
        "169FTA",
        "AB1CDE",        # right length, wrong pattern
        "",              # empty
        "AB12CD",        # short
        "AB12CDEF",      # long
    ],
)
def test_is_canonical_uk_plate_rejects_non_uk_shapes(plate: str) -> None:
    assert is_canonical_uk_plate(plate) is False


# ----------------------------------------------------------------------
# DvsaClient.lookup_plate happy path


def test_lookup_plate_returns_trimmed_result(dvsa_config: DvsaConfig) -> None:
    """The client returns only the fields the labelling pass needs --
    not the full motTests history. Locks down the trim against the
    real Porsche body."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.return_value = _mot_response(_REAL_PORSCHE_BODY)

    client = DvsaClient(dvsa_config, session=session)
    result = client.lookup_plate("AE13SJX")

    assert isinstance(result, MotLookupResult)
    assert result.registration == "AE13SJX"
    assert result.make == "PORSCHE"
    assert result.model == "911"
    assert result.year == 2013
    assert result.primary_colour == "Black"
    assert result.fuel_type == "Petrol"
    assert result.engine_size_cc == 3800
    # Just check shape; the exact value moves second-to-second.
    assert "T" in result.looked_up_at
    assert result.looked_up_at.endswith("+00:00")


def test_lookup_plate_normalises_registration(dvsa_config: DvsaConfig) -> None:
    """Operator-typed plates may have spaces / lowercase. Upper +
    strip-spaces before hitting the API."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.return_value = _mot_response(_REAL_PORSCHE_BODY)

    client = DvsaClient(dvsa_config, session=session)
    client.lookup_plate("ae13 sjx")

    called_url = session.get.call_args.args[0]
    assert called_url.endswith("/v1/trade/vehicles/registration/AE13SJX")


def test_lookup_plate_empty_string_returns_none(dvsa_config: DvsaConfig) -> None:
    """Don't waste an API call on an empty plate."""
    session = MagicMock(spec=requests.Session)
    client = DvsaClient(dvsa_config, session=session)
    assert client.lookup_plate("") is None
    assert client.lookup_plate("   ") is None
    # No HTTP call should have happened.
    assert session.post.call_count == 0
    assert session.get.call_count == 0


# ----------------------------------------------------------------------
# 404 / 401 / 429 handling


def test_lookup_plate_returns_none_on_404(dvsa_config: DvsaConfig) -> None:
    """Plate not on the MOT register (vehicle < 3 years old, exported,
    or DVSA simply has no record). Caller logs and skips."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.return_value = _err_response(404)

    client = DvsaClient(dvsa_config, session=session)
    result = client.lookup_plate("XX00XXX")
    assert result is None


def test_lookup_plate_refreshes_token_on_401_and_retries(
    dvsa_config: DvsaConfig,
) -> None:
    """A revoked / mid-run-rotated token returns 401. The client
    forces a refresh and retries once."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.side_effect = [
        _err_response(401),
        _mot_response(_REAL_PORSCHE_BODY),
    ]

    client = DvsaClient(dvsa_config, session=session)
    result = client.lookup_plate("AE13SJX")

    assert result is not None
    assert result.make == "PORSCHE"
    # Two token POSTs: initial + forced refresh after the 401.
    assert session.post.call_count == 2
    assert session.get.call_count == 2


def test_lookup_plate_backs_off_and_retries_on_429(
    dvsa_config: DvsaConfig,
) -> None:
    """429 with a Retry-After header: sleep then retry once. We patch
    time.sleep so the test isn't actually slow."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.side_effect = [
        _err_response(429, retry_after="0.5"),
        _mot_response(_REAL_PORSCHE_BODY),
    ]

    client = DvsaClient(dvsa_config, session=session)
    with patch("streettracker.analysis.dvsa.time.sleep") as mock_sleep:
        result = client.lookup_plate("AE13SJX")

    assert result is not None
    # The 429-backoff sleep call (separate from the rate-limit throttle).
    assert any(c.args[0] == pytest.approx(0.5) for c in mock_sleep.call_args_list)
    assert session.get.call_count == 2


def test_lookup_plate_raises_on_5xx(dvsa_config: DvsaConfig) -> None:
    """500-class responses are not silently ignored -- a broken upstream
    needs to surface to the operator, not produce empty labels."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response()
    session.get.return_value = _err_response(503)

    client = DvsaClient(dvsa_config, session=session)
    with pytest.raises(requests.HTTPError):
        client.lookup_plate("AE13SJX")


# ----------------------------------------------------------------------
# Token reuse


def test_token_is_reused_across_calls(dvsa_config: DvsaConfig) -> None:
    """A fresh token (1h TTL) should serve every lookup in a batch;
    the client must not POST to the token endpoint for each plate."""
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _token_response(expires_in=3600.0)
    session.get.return_value = _mot_response(_REAL_PORSCHE_BODY)

    client = DvsaClient(dvsa_config, session=session)
    for _ in range(5):
        client.lookup_plate("AE13SJX")

    assert session.post.call_count == 1
    assert session.get.call_count == 5


def test_token_refreshed_when_near_expiry(dvsa_config: DvsaConfig) -> None:
    """A token within the refresh buffer (60 s) of expiry is considered
    stale and replaced before the next call."""
    # First response: nearly-expired token.
    short_resp = _token_response(expires_in=10.0)
    fresh_resp = _token_response(expires_in=3600.0)
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = [short_resp, fresh_resp]
    session.get.return_value = _mot_response(_REAL_PORSCHE_BODY)

    client = DvsaClient(dvsa_config, session=session)
    client.lookup_plate("AE13SJX")  # uses the 10-s token
    client.lookup_plate("AE13SJX")  # 10 s < 60 s buffer -> refresh

    assert session.post.call_count == 2
