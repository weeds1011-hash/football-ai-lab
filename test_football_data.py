from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from models import analyze_dataframe
from football_data import (
    FootballDataAuthenticationError,
    FootballDataClient,
    FootballDataFormatError,
    FootballDataRateLimitError,
    build_real_match_bundle,
    season_start_year,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def _team(team_id: int, name: str) -> dict[str, object]:
    return {"id": team_id, "name": name, "shortName": name, "tla": name[:3]}


def _finished_match(
    match_id: int,
    kickoff: datetime,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, object]:
    return {
        "id": match_id,
        "utcDate": kickoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "FINISHED",
        "homeTeam": _team(home_id, home_name),
        "awayTeam": _team(away_id, away_name),
        "score": {"fullTime": {"home": home_goals, "away": away_goals}},
    }


def _scheduled_match(
    match_id: int,
    kickoff: datetime,
    home_id: int = 1,
    home_name: str = "실제 홈",
    away_id: int = 2,
    away_name: str = "실제 원정",
) -> dict[str, object]:
    return {
        "id": match_id,
        "utcDate": kickoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "TIMED",
        "homeTeam": _team(home_id, home_name),
        "awayTeam": _team(away_id, away_name),
        "score": {"fullTime": {"home": None, "away": None}},
    }


def _history(fixture_time: datetime) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for index in range(12):
        kickoff = fixture_time - timedelta(days=30 - index)
        opponent_id = 100 + index
        if index % 2 == 0:
            matches.append(
                _finished_match(
                    index + 1,
                    kickoff,
                    1,
                    "실제 홈",
                    opponent_id,
                    f"상대 {index}",
                    2 if index % 4 == 0 else 1,
                    0,
                )
            )
        else:
            matches.append(
                _finished_match(
                    index + 1,
                    kickoff,
                    opponent_id,
                    f"상대 {index}",
                    2,
                    "실제 원정",
                    1,
                    1 if index % 3 == 0 else 2,
                )
            )
    return matches


def test_client_sends_token_season_and_reads_rate_headers() -> None:
    response = FakeResponse(
        200,
        {"matches": []},
        headers={
            "X-Requests-Available-Minute": "8",
            "X-RequestCounter-Reset": "17",
        },
    )
    session = FakeSession(response)
    result = FootballDataClient("test-token", session=session).competition_matches(
        "PL", 2026
    )

    assert session.calls == [
        {
            "url": "https://api.football-data.org/v4/competitions/PL/matches",
            "headers": {"X-Auth-Token": "test-token"},
            "params": {"season": 2026},
            "timeout": 15.0,
        }
    ]
    assert result.rate_limit.remaining_per_minute == 8
    assert result.rate_limit.reset_seconds == 17


def test_client_reports_rate_limit_and_retry_seconds() -> None:
    session = FakeSession(
        FakeResponse(
            429,
            {"message": "too many requests"},
            headers={"Retry-After": "31"},
        )
    )
    with pytest.raises(FootballDataRateLimitError) as raised:
        FootballDataClient("test-token", session=session).competition_matches("PL", 2026)
    assert raised.value.retry_after_seconds == 31


def test_client_reports_authentication_and_missing_matches() -> None:
    forbidden = FakeSession(FakeResponse(403, {"message": "restricted"}))
    with pytest.raises(FootballDataAuthenticationError, match="접근 범위"):
        FootballDataClient("bad-token", session=forbidden).competition_matches("PL", 2026)

    malformed = FakeSession(FakeResponse(200, {"competition": {"code": "PL"}}))
    with pytest.raises(FootballDataFormatError, match="matches 배열"):
        FootballDataClient("test-token", session=malformed).competition_matches("PL", 2026)


def test_real_bundle_uses_only_results_before_fixture_and_marks_estimated_xg() -> None:
    fixture_time = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
    history = _history(fixture_time)
    payload = {
        "competition": {"name": "Premier League", "code": "PL"},
        "matches": history + [_scheduled_match(500, fixture_time)],
    }
    now = datetime(2026, 10, 27, 0, tzinfo=timezone.utc)
    baseline = build_real_match_bundle(
        payload, target_date=date(2026, 10, 27), now=now
    )

    future_result = _finished_match(
        999,
        fixture_time + timedelta(hours=2),
        1,
        "실제 홈",
        2,
        "실제 원정",
        9,
        0,
    )
    with_future = build_real_match_bundle(
        {**payload, "matches": payload["matches"] + [future_result]},
        target_date=date(2026, 10, 27),
        now=now,
    )

    pd.testing.assert_frame_equal(baseline.input_frame, with_future.input_frame)
    assert len(baseline.input_frame) == 1
    assert baseline.input_frame.loc[0, "데이터 구분"] == "실제 일정·결과 / 모델 추정 기대득점"
    assert "실측 xG 아님" in baseline.evidence_frame.loc[0, "기대득점 구분"]
    assert baseline.evidence_frame.loc[0, "홈 최근5"].endswith("(5경기)")
    assert "(6경기)" in baseline.evidence_frame.loc[0, "홈팀 홈/원정 성적"]
    assert "(6경기)" in baseline.evidence_frame.loc[0, "원정팀 홈/원정 성적"]
    assert pd.isna(baseline.input_frame.loc[0, "홈승 배당"])
    _, analysis_output, _ = analyze_dataframe(
        baseline.input_frame,
        default_data_type="실제 일정·결과 / 모델 추정 기대득점",
    )
    probability_columns = ["홈승 확률(%)", "무승부 확률(%)", "원정승 확률(%)"]
    assert analysis_output.loc[0, probability_columns].sum() == pytest.approx(100.0)


def test_missing_score_is_skipped_and_sparse_history_has_high_uncertainty() -> None:
    fixture_time = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
    missing_score = _finished_match(
        1, fixture_time - timedelta(days=1), 1, "실제 홈", 3, "상대", 1, 0
    )
    missing_score["score"] = {"fullTime": {"home": None, "away": None}}
    bundle = build_real_match_bundle(
        {
            "competition": {"name": "Premier League"},
            "matches": [missing_score, _scheduled_match(2, fixture_time)],
        },
        target_date=date(2026, 10, 27),
        now=datetime(2026, 10, 27, 0, tzinfo=timezone.utc),
    )
    assert bundle.skipped_match_count == 1
    assert bundle.evidence_frame.loc[0, "불확실성"] == "높음"
    assert bundle.input_frame.loc[0, "홈 기대득점"] == pytest.approx(1.35)
    assert bundle.input_frame.loc[0, "원정 기대득점"] == pytest.approx(1.10)


def test_started_fixture_is_not_returned_for_live_prediction() -> None:
    fixture_time = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
    bundle = build_real_match_bundle(
        {"matches": [_scheduled_match(1, fixture_time)]},
        target_date=date(2026, 10, 27),
        now=fixture_time + timedelta(minutes=1),
    )
    assert bundle.input_frame.empty


def test_season_start_year_handles_european_and_calendar_competitions() -> None:
    assert season_start_year(date(2026, 1, 10), "PL") == 2025
    assert season_start_year(date(2026, 8, 10), "PL") == 2026
    assert season_start_year(date(2026, 1, 10), "BSA") == 2026
