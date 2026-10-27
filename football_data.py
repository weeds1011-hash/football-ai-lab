"""football-data.org v4 연동과 과거 결과 기반 팀 전력 추정.

API가 제공하는 실제 일정/결과와 앱이 계산한 기대득점을 의도적으로 분리한다.
예측 대상 경기의 킥오프보다 이른 FINISHED 경기만 사용해 미래 정보 누수를 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests


API_BASE_URL = "https://api.football-data.org/v4"
SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")
FINISHED_STATUS = "FINISHED"
UPCOMING_STATUSES = frozenset({"SCHEDULED", "TIMED"})


@dataclass(frozen=True)
class Competition:
    code: str
    korean_name: str
    area: str
    calendar_year_season: bool = False

    @property
    def label(self) -> str:
        return f"{self.korean_name} ({self.code})"


# football-data.org가 무료 제공 대상으로 명시한 대회 코드다.
FREE_COMPETITIONS: tuple[Competition, ...] = (
    Competition("PL", "프리미어리그", "잉글랜드"),
    Competition("ELC", "챔피언십", "잉글랜드"),
    Competition("BL1", "분데스리가", "독일"),
    Competition("PD", "라리가", "스페인"),
    Competition("SA", "세리에 A", "이탈리아"),
    Competition("FL1", "리그 1", "프랑스"),
    Competition("DED", "에레디비시", "네덜란드"),
    Competition("PPL", "프리메이라 리가", "포르투갈"),
    Competition("CL", "UEFA 챔피언스리그", "유럽"),
    Competition("BSA", "브라질 세리에 A", "브라질", True),
    Competition("WC", "FIFA 월드컵", "세계", True),
    Competition("EC", "UEFA 유럽선수권", "유럽", True),
)


class FootballDataError(RuntimeError):
    """사용자에게 표시할 수 있는 football-data.org 오류."""


class FootballDataAuthenticationError(FootballDataError):
    """API 키 누락 또는 접근 권한 오류."""


class FootballDataRateLimitError(FootballDataError):
    """요청 한도 초과 오류."""

    def __init__(self, message: str, *, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class FootballDataFormatError(FootballDataError):
    """필수 필드가 없는 API 응답 오류."""


@dataclass(frozen=True)
class RateLimitInfo:
    remaining_per_minute: int | None
    reset_seconds: int | None


@dataclass(frozen=True)
class ApiResponse:
    data: Mapping[str, Any]
    rate_limit: RateLimitInfo


@dataclass(frozen=True)
class HistoricalMatch:
    match_id: str
    kickoff: datetime
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class TeamForm:
    team_id: int
    team_name: str
    elo: float
    matches_available: int
    recent5_record: str
    recent10_record: str
    recent10_goals_for: float | None
    recent10_goals_against: float | None
    home_record: str
    away_record: str
    venue_matches: int
    venue_goals_for: float | None
    venue_goals_against: float | None
    uncertainty: str


@dataclass(frozen=True)
class RealMatchBundle:
    input_frame: pd.DataFrame
    evidence_frame: pd.DataFrame
    history_match_count: int
    skipped_match_count: int


class FootballDataClient:
    """작고 테스트 가능한 football-data.org v4 HTTP 클라이언트."""

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | Any | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        token = api_key.strip()
        if not token:
            raise FootballDataAuthenticationError("football-data.org API 키가 없습니다.")
        self.api_key = token
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def get_json(
        self, path: str, *, params: Mapping[str, object] | None = None
    ) -> ApiResponse:
        url = f"{API_BASE_URL}/{path.lstrip('/')}"
        try:
            response = self.session.get(
                url,
                headers={"X-Auth-Token": self.api_key},
                params=dict(params or {}),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise FootballDataError(
                "football-data.org에 연결할 수 없습니다. 잠시 후 재시도하거나 CSV를 사용하세요."
            ) from exc

        rate_limit = RateLimitInfo(
            remaining_per_minute=_header_int(
                response.headers, "X-Requests-Available-Minute"
            ),
            reset_seconds=_header_int(response.headers, "X-RequestCounter-Reset"),
        )
        error_message = _response_error_message(response)
        if response.status_code == 429:
            retry_after = _header_int(response.headers, "Retry-After")
            if retry_after is None:
                retry_after = rate_limit.reset_seconds
            raise FootballDataRateLimitError(
                "football-data.org 요청 한도를 초과했습니다. 캐시 만료 후 다시 시도하세요.",
                retry_after_seconds=retry_after,
            )
        if response.status_code in {401, 403}:
            raise FootballDataAuthenticationError(
                f"API 키 또는 무료 플랜 접근 범위를 확인하세요. {error_message}".strip()
            )
        if response.status_code >= 400:
            raise FootballDataError(
                f"football-data.org 오류({response.status_code}): {error_message}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise FootballDataFormatError("API 응답이 올바른 JSON이 아닙니다.") from exc
        if not isinstance(payload, Mapping):
            raise FootballDataFormatError("API 응답의 최상위 형식이 객체가 아닙니다.")
        return ApiResponse(data=payload, rate_limit=rate_limit)

    def competition_matches(self, competition_code: str, season: int) -> ApiResponse:
        response = self.get_json(
            f"competitions/{competition_code}/matches", params={"season": season}
        )
        if not isinstance(response.data.get("matches"), list):
            raise FootballDataFormatError("API 응답에 matches 배열이 없습니다.")
        return response

    def competition_teams(self, competition_code: str, season: int) -> ApiResponse:
        response = self.get_json(
            f"competitions/{competition_code}/teams", params={"season": season}
        )
        if not isinstance(response.data.get("teams"), list):
            raise FootballDataFormatError("API 응답에 teams 배열이 없습니다.")
        return response


def season_start_year(target_date: date, competition_code: str) -> int:
    competition = next(
        (item for item in FREE_COMPETITIONS if item.code == competition_code), None
    )
    if competition is not None and competition.calendar_year_season:
        return target_date.year
    return target_date.year if target_date.month >= 7 else target_date.year - 1


def build_real_match_bundle(
    payload: Mapping[str, Any],
    *,
    target_date: date,
    now: datetime | None = None,
    local_timezone: ZoneInfo = SEOUL_TIMEZONE,
) -> RealMatchBundle:
    """실제 일정과 과거 결과를 기존 분석 입력 및 근거표로 변환한다.

    예측 경기마다 그 경기의 kickoff보다 앞선 완료 경기만 사용한다. 현재보다 이미
    시작한 경기는 실전 예측 목록에서 제외한다.
    """

    raw_matches = payload.get("matches")
    if not isinstance(raw_matches, list):
        raise FootballDataFormatError("API 응답에 matches 배열이 없습니다.")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now는 시간대 정보가 있는 datetime이어야 합니다.")

    historical: list[HistoricalMatch] = []
    fixtures: list[tuple[Mapping[str, Any], datetime]] = []
    skipped = 0
    for item in raw_matches:
        if not isinstance(item, Mapping):
            skipped += 1
            continue
        try:
            kickoff = _parse_utc_datetime(item.get("utcDate"))
        except FootballDataFormatError:
            skipped += 1
            continue

        status = str(item.get("status") or "")
        if status == FINISHED_STATUS:
            parsed = _parse_finished_match(item, kickoff)
            if parsed is None:
                skipped += 1
            else:
                historical.append(parsed)
            continue

        local_date = kickoff.astimezone(local_timezone).date()
        if (
            status in UPCOMING_STATUSES
            and local_date == target_date
            and kickoff > current_time.astimezone(timezone.utc)
        ):
            fixtures.append((item, kickoff))

    historical.sort(key=lambda match: match.kickoff)
    fixtures.sort(key=lambda item: item[1])

    input_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    competition_name = _nested_string(payload, "competition", "name") or "선택 리그"
    competition_code = _nested_string(payload, "competition", "code") or ""

    for raw_fixture, kickoff in fixtures:
        teams = _fixture_teams(raw_fixture)
        if teams is None:
            skipped += 1
            continue
        home_id, home_name, away_id, away_name = teams
        prior_matches = [match for match in historical if match.kickoff < kickoff]
        ratings = calculate_elo_ratings(prior_matches)
        home_form = calculate_team_form(
            prior_matches,
            team_id=home_id,
            team_name=home_name,
            elo=ratings.get(home_id, 1500.0),
            prediction_venue="home",
        )
        away_form = calculate_team_form(
            prior_matches,
            team_id=away_id,
            team_name=away_name,
            elo=ratings.get(away_id, 1500.0),
            prediction_venue="away",
        )
        home_xg, away_xg = estimate_expected_goals(
            prior_matches, home_id=home_id, away_id=away_id
        )
        uncertainty = _combined_uncertainty(home_form, away_form)
        match_id = str(raw_fixture.get("id") or f"{home_id}-{away_id}-{kickoff.isoformat()}")
        local_kickoff = kickoff.astimezone(local_timezone)

        input_rows.append(
            {
                "경기번호": match_id,
                "경기시각(KST)": local_kickoff.strftime("%Y-%m-%d %H:%M"),
                "대회코드": competition_code,
                "홈팀 ID": home_id,
                "원정팀 ID": away_id,
                "홈팀": home_name,
                "원정팀": away_name,
                "홈 기대득점": round(home_xg, 3),
                "원정 기대득점": round(away_xg, 3),
                "홈 Elo": round(home_form.elo, 1),
                "원정 Elo": round(away_form.elo, 1),
                "홈승 배당": math.nan,
                "무승부 배당": math.nan,
                "원정승 배당": math.nan,
                "데이터 구분": "실제 일정·결과 / 모델 추정 기대득점",
            }
        )
        evidence_rows.append(
            {
                "경기번호": match_id,
                "경기시각(KST)": local_kickoff.strftime("%Y-%m-%d %H:%M"),
                "리그": competition_name,
                "경기": f"{home_name} vs {away_name}",
                "홈 최근5": home_form.recent5_record,
                "원정 최근5": away_form.recent5_record,
                "홈 최근10": home_form.recent10_record,
                "원정 최근10": away_form.recent10_record,
                "홈 최근10 득점/실점": _goal_average_label(home_form),
                "원정 최근10 득점/실점": _goal_average_label(away_form),
                "홈팀 홈/원정 성적": f"{home_form.home_record} / {home_form.away_record}",
                "원정팀 홈/원정 성적": f"{away_form.home_record} / {away_form.away_record}",
                "홈 Elo": round(home_form.elo, 1),
                "원정 Elo": round(away_form.elo, 1),
                "추정 기대득점": f"{home_xg:.2f} - {away_xg:.2f}",
                "불확실성": uncertainty,
                "기대득점 구분": "경기 결과 기반 모델 추정치 (실측 xG 아님)",
            }
        )

    return RealMatchBundle(
        input_frame=pd.DataFrame(input_rows),
        evidence_frame=pd.DataFrame(evidence_rows),
        history_match_count=len(historical),
        skipped_match_count=skipped,
    )


def calculate_elo_ratings(
    matches: Sequence[HistoricalMatch], *, k_factor: float = 20.0
) -> dict[int, float]:
    ratings: dict[int, float] = {}
    for match in sorted(matches, key=lambda item: item.kickoff):
        home_rating = ratings.get(match.home_team_id, 1500.0)
        away_rating = ratings.get(match.away_team_id, 1500.0)
        expected_home = 1.0 / (1.0 + 10.0 ** (-(home_rating + 65.0 - away_rating) / 400.0))
        if match.home_goals > match.away_goals:
            actual_home = 1.0
        elif match.home_goals == match.away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0
        change = k_factor * (actual_home - expected_home)
        ratings[match.home_team_id] = home_rating + change
        ratings[match.away_team_id] = away_rating - change
    return ratings


def calculate_team_form(
    matches: Sequence[HistoricalMatch],
    *,
    team_id: int,
    team_name: str,
    elo: float,
    prediction_venue: str,
) -> TeamForm:
    events = _team_events(matches, team_id)
    recent5 = events[-5:]
    recent10 = events[-10:]
    home_events = [event for event in events if event[0] == "home"][-10:]
    away_events = [event for event in events if event[0] == "away"][-10:]
    venue_events = home_events if prediction_venue == "home" else away_events
    if len(events) < 5 or len(venue_events) < 3:
        uncertainty = "높음"
    elif len(events) < 10 or len(venue_events) < 5:
        uncertainty = "보통"
    else:
        uncertainty = "낮음"
    return TeamForm(
        team_id=team_id,
        team_name=team_name,
        elo=elo,
        matches_available=len(events),
        recent5_record=_record_label(recent5),
        recent10_record=_record_label(recent10),
        recent10_goals_for=_average(event[1] for event in recent10),
        recent10_goals_against=_average(event[2] for event in recent10),
        home_record=_record_label(home_events),
        away_record=_record_label(away_events),
        venue_matches=len(venue_events),
        venue_goals_for=_average(event[1] for event in venue_events),
        venue_goals_against=_average(event[2] for event in venue_events),
        uncertainty=uncertainty,
    )


def estimate_expected_goals(
    matches: Sequence[HistoricalMatch], *, home_id: int, away_id: int
) -> tuple[float, float]:
    """홈/원정 득실점률을 리그 평균으로 수축해 기대득점을 추정한다."""

    league_home = _average(match.home_goals for match in matches) or 1.35
    league_away = _average(match.away_goals for match in matches) or 1.10
    home_events = [
        event for event in _team_events(matches, home_id) if event[0] == "home"
    ][-10:]
    away_events = [
        event for event in _team_events(matches, away_id) if event[0] == "away"
    ][-10:]

    home_scored = _shrunk_average(home_events, 1, league_home)
    home_conceded = _shrunk_average(home_events, 2, league_away)
    away_scored = _shrunk_average(away_events, 1, league_away)
    away_conceded = _shrunk_average(away_events, 2, league_home)

    home_expected = league_home * (home_scored / league_home) * (away_conceded / league_home)
    away_expected = league_away * (away_scored / league_away) * (home_conceded / league_away)
    return _clamp(home_expected, 0.2, 4.5), _clamp(away_expected, 0.2, 4.5)


def _parse_finished_match(
    item: Mapping[str, Any], kickoff: datetime
) -> HistoricalMatch | None:
    teams = _fixture_teams(item)
    if teams is None:
        return None
    score = item.get("score")
    full_time = score.get("fullTime") if isinstance(score, Mapping) else None
    if not isinstance(full_time, Mapping):
        return None
    home_goals = full_time.get("home")
    away_goals = full_time.get("away")
    if not isinstance(home_goals, int) or not isinstance(away_goals, int):
        return None
    home_id, home_name, away_id, away_name = teams
    return HistoricalMatch(
        match_id=str(item.get("id") or ""),
        kickoff=kickoff,
        home_team_id=home_id,
        home_team=home_name,
        away_team_id=away_id,
        away_team=away_name,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def _fixture_teams(
    item: Mapping[str, Any],
) -> tuple[int, str, int, str] | None:
    home = item.get("homeTeam")
    away = item.get("awayTeam")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        return None
    home_id, away_id = home.get("id"), away.get("id")
    home_name, away_name = home.get("name"), away.get("name")
    if (
        not isinstance(home_id, int)
        or not isinstance(away_id, int)
        or not isinstance(home_name, str)
        or not isinstance(away_name, str)
        or not home_name.strip()
        or not away_name.strip()
    ):
        return None
    return home_id, home_name.strip(), away_id, away_name.strip()


def _team_events(
    matches: Sequence[HistoricalMatch], team_id: int
) -> list[tuple[str, int, int]]:
    events: list[tuple[str, int, int]] = []
    for match in sorted(matches, key=lambda item: item.kickoff):
        if match.home_team_id == team_id:
            events.append(("home", match.home_goals, match.away_goals))
        elif match.away_team_id == team_id:
            events.append(("away", match.away_goals, match.home_goals))
    return events


def _record_label(events: Sequence[tuple[str, int, int]]) -> str:
    wins = sum(goals_for > goals_against for _, goals_for, goals_against in events)
    draws = sum(goals_for == goals_against for _, goals_for, goals_against in events)
    losses = len(events) - wins - draws
    return f"{wins}승 {draws}무 {losses}패 ({len(events)}경기)"


def _goal_average_label(form: TeamForm) -> str:
    if form.recent10_goals_for is None or form.recent10_goals_against is None:
        return "자료 없음"
    return f"{form.recent10_goals_for:.2f} / {form.recent10_goals_against:.2f}"


def _combined_uncertainty(home: TeamForm, away: TeamForm) -> str:
    order = {"낮음": 0, "보통": 1, "높음": 2}
    return max((home.uncertainty, away.uncertainty), key=order.__getitem__)


def _shrunk_average(
    events: Sequence[tuple[str, int, int]], index: int, baseline: float, prior: int = 5
) -> float:
    observed_total = sum(event[index] for event in events)
    return (observed_total + baseline * prior) / (len(events) + prior)


def _average(values: Any) -> float | None:
    items = list(values)
    return None if not items else float(sum(items) / len(items))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _parse_utc_datetime(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise FootballDataFormatError("경기 utcDate가 없습니다.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FootballDataFormatError("경기 utcDate 형식이 올바르지 않습니다.") from exc
    if parsed.tzinfo is None:
        raise FootballDataFormatError("경기 utcDate에 시간대가 없습니다.")
    return parsed.astimezone(timezone.utc)


def _nested_string(payload: Mapping[str, Any], parent: str, child: str) -> str | None:
    nested = payload.get(parent)
    value = nested.get(child) if isinstance(nested, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _header_int(headers: Mapping[str, Any], name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_error_message(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        return str(getattr(response, "text", "")).strip()
    if isinstance(payload, Mapping):
        message = payload.get("message") or payload.get("error")
        if message is not None:
            return str(message).strip()
    return ""
