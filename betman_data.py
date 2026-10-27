"""사용자 제공 베트맨 자료를 안전하게 정규화하고 경기 모델에 연결한다.

베트맨 사이트를 자동 수집하지 않는다. 이 모듈은 사용자가 이용 권한을 확인해
확보한 CSV/XLSX만 처리하며, 팀명은 명시적 별칭 또는 보수적인 완전 일치로만
football-data.org 경기와 연결한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import re
import unicodedata
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")

BETMAN_HOME_URL = "https://www.betman.co.kr/"
BETMAN_TERMS_URL = (
    "https://www.betman.co.kr/main/mainPage/customercenter/agreement.do"
)
BETMAN_SOURCE_LABEL = "베트맨 사용자 제공 자료"

MARKET_1X2 = "일반 승무패"
MARKET_HANDICAP = "핸디캡"
MARKET_TOTALS = "언더오버"
MARKET_OTHER = "기타 시장"

BETMAN_TEMPLATE_COLUMNS = (
    "상품",
    "회차",
    "경기번호",
    "종목",
    "대회",
    "대회코드",
    "경기시각(KST)",
    "마감시각(KST)",
    "홈팀",
    "원정팀",
    "시장유형",
    "홈승 배당",
    "무승부 배당",
    "원정승 배당",
    "승 투표율(%)",
    "무 투표율(%)",
    "패 투표율(%)",
    "홈팀 ID",
    "원정팀 ID",
    "조회기준시각(KST)",
    "출처 URL",
)

TEAM_ALIAS_TEMPLATE_COLUMNS = (
    "대회코드",
    "베트맨 팀명",
    "football-data 팀명",
)

_COLUMN_ALIASES = {
    "상품": "product",
    "product": "product",
    "게임": "product",
    "회차": "round_id",
    "round": "round_id",
    "round_id": "round_id",
    "경기번호": "match_no",
    "경기_번호": "match_no",
    "match_no": "match_no",
    "match_number": "match_no",
    "종목": "sport",
    "sport": "sport",
    "대회": "league",
    "리그": "league",
    "league": "league",
    "대회코드": "competition_code",
    "대회_코드": "competition_code",
    "competition_code": "competition_code",
    "경기시각(kst)": "kickoff_kst",
    "경기시각": "kickoff_kst",
    "개최시간": "kickoff_kst",
    "kickoff_kst": "kickoff_kst",
    "kickoff": "kickoff_kst",
    "마감시각(kst)": "deadline_kst",
    "마감시각": "deadline_kst",
    "마감시간": "deadline_kst",
    "deadline_kst": "deadline_kst",
    "홈팀": "home_team",
    "home_team": "home_team",
    "원정팀": "away_team",
    "away_team": "away_team",
    "시장유형": "market_raw",
    "시장_유형": "market_raw",
    "게임유형": "market_raw",
    "game_type": "market_raw",
    "market": "market_raw",
    "홈승배당": "home_odds",
    "홈승_배당": "home_odds",
    "홈승 배당": "home_odds",
    "home_odds": "home_odds",
    "무승부배당": "draw_odds",
    "무승부_배당": "draw_odds",
    "무승부 배당": "draw_odds",
    "draw_odds": "draw_odds",
    "원정승배당": "away_odds",
    "원정승_배당": "away_odds",
    "원정승 배당": "away_odds",
    "away_odds": "away_odds",
    "승투표율(%)": "home_vote_pct",
    "승_투표율(%)": "home_vote_pct",
    "승 투표율(%)": "home_vote_pct",
    "home_vote_pct": "home_vote_pct",
    "무투표율(%)": "draw_vote_pct",
    "무_투표율(%)": "draw_vote_pct",
    "무 투표율(%)": "draw_vote_pct",
    "draw_vote_pct": "draw_vote_pct",
    "패투표율(%)": "away_vote_pct",
    "패_투표율(%)": "away_vote_pct",
    "패 투표율(%)": "away_vote_pct",
    "away_vote_pct": "away_vote_pct",
    "홈팀id": "home_team_id",
    "홈팀_id": "home_team_id",
    "홈팀 ID": "home_team_id",
    "home_team_id": "home_team_id",
    "원정팀id": "away_team_id",
    "원정팀_id": "away_team_id",
    "원정팀 ID": "away_team_id",
    "away_team_id": "away_team_id",
    "조회기준시각(kst)": "retrieved_at_kst",
    "조회기준시각": "retrieved_at_kst",
    "조회 기준시각": "retrieved_at_kst",
    "retrieved_at_kst": "retrieved_at_kst",
    "출처url": "source_url",
    "출처_url": "source_url",
    "출처 URL": "source_url",
    "source_url": "source_url",
}

_REQUIRED_COLUMNS = (
    "round_id",
    "match_no",
    "sport",
    "kickoff_kst",
    "home_team",
    "away_team",
    "market_raw",
)


@dataclass(frozen=True)
class BetmanLinkResult:
    """베트맨 경기와 football-data.org 경기 연결 결과."""

    model_frame: pd.DataFrame
    status_frame: pd.DataFrame

    @property
    def matched_count(self) -> int:
        return len(self.model_frame)


def betman_template_csv() -> bytes:
    """실제 값을 포함하지 않는 빈 입력 템플릿을 반환한다."""

    return (
        pd.DataFrame(columns=BETMAN_TEMPLATE_COLUMNS)
        .to_csv(index=False)
        .encode("utf-8-sig")
    )


def team_alias_template_csv() -> bytes:
    """팀 별칭 수동 확인용 빈 템플릿을 반환한다."""

    return (
        pd.DataFrame(columns=TEAM_ALIAS_TEMPLATE_COLUMNS)
        .to_csv(index=False)
        .encode("utf-8-sig")
    )


def read_betman_file(payload: bytes, filename: str) -> pd.DataFrame:
    """사용자 제공 CSV 또는 XLSX를 표준 베트맨 스키마로 읽는다."""

    frame = _read_table_file(payload, filename, label="베트맨 경기 파일")
    return normalise_betman_dataframe(frame)


def read_team_alias_file(payload: bytes, filename: str) -> pd.DataFrame:
    """명시적으로 확인된 팀명 매핑 CSV/XLSX를 읽는다."""

    frame = _read_table_file(payload, filename, label="팀명 매핑 파일")
    aliases = {
        _normalise_header("대회코드"): "competition_code",
        _normalise_header("competition_code"): "competition_code",
        _normalise_header("베트맨 팀명"): "source_team",
        _normalise_header("source_team"): "source_team",
        _normalise_header("football-data 팀명"): "target_team",
        _normalise_header("football_data_team"): "target_team",
        _normalise_header("target_team"): "target_team",
    }
    renamed: dict[object, str] = {}
    for column in frame.columns:
        target = aliases.get(_normalise_header(column))
        if target:
            renamed[column] = target
    result = frame.rename(columns=renamed).copy()
    missing = [name for name in ("source_team", "target_team") if name not in result]
    if missing:
        raise ValueError("팀명 매핑 필수 열이 없습니다: " + ", ".join(missing))
    if "competition_code" not in result:
        result["competition_code"] = ""
    for column in ("competition_code", "source_team", "target_team"):
        result[column] = result[column].astype("string").fillna("").str.strip()
    if (result["source_team"] == "").any() or (result["target_team"] == "").any():
        raise ValueError("팀명 매핑에는 빈 팀 이름을 사용할 수 없습니다.")
    return result.loc[:, ["competition_code", "source_team", "target_team"]].reset_index(
        drop=True
    )


def normalise_betman_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """한국어/영문 열을 정규화하고 회차 자료를 검증한다."""

    if frame.empty:
        raise ValueError("베트맨 경기 파일에 데이터 행이 없습니다.")

    rename_map: dict[object, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        target = _COLUMN_ALIASES.get(_normalise_header(column))
        if target is None:
            continue
        if target in seen:
            raise ValueError(f"같은 의미의 열이 중복되었습니다: {target}")
        seen.add(target)
        rename_map[column] = target

    result = frame.rename(columns=rename_map).copy()
    missing = [column for column in _REQUIRED_COLUMNS if column not in result]
    if missing:
        raise ValueError("베트맨 필수 열이 없습니다: " + ", ".join(missing))

    defaults: Mapping[str, object] = {
        "product": "",
        "league": "",
        "competition_code": "",
        "deadline_kst": pd.NaT,
        "home_odds": math.nan,
        "draw_odds": math.nan,
        "away_odds": math.nan,
        "home_vote_pct": math.nan,
        "draw_vote_pct": math.nan,
        "away_vote_pct": math.nan,
        "home_team_id": pd.NA,
        "away_team_id": pd.NA,
        "retrieved_at_kst": pd.NaT,
        "source_url": "",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default

    for column in (
        "product",
        "round_id",
        "sport",
        "league",
        "competition_code",
        "home_team",
        "away_team",
        "market_raw",
        "source_url",
    ):
        result[column] = result[column].astype("string").fillna("").str.strip()

    for column in ("round_id", "sport", "home_team", "away_team", "market_raw"):
        if (result[column] == "").any():
            raise ValueError(f"{column} 열에 빈 값이 있습니다.")

    result["match_no"] = result["match_no"].map(_parse_match_number)
    if result["match_no"].isna().any():
        raise ValueError("경기번호는 1 이상의 숫자여야 합니다.")
    result["match_no"] = result["match_no"].astype(int)

    result["kickoff_kst"] = result["kickoff_kst"].map(
        lambda value: _parse_kst_datetime(value, required=True, label="경기시각")
    )
    result["deadline_kst"] = result["deadline_kst"].map(
        lambda value: _parse_kst_datetime(value, required=False, label="마감시각")
    )
    result["retrieved_at_kst"] = result["retrieved_at_kst"].map(
        lambda value: _parse_kst_datetime(value, required=False, label="조회기준시각")
    )
    result["market_type"] = result["market_raw"].map(classify_market)

    for column in ("home_odds", "draw_odds", "away_odds"):
        result[column] = _optional_numeric(result[column], column)
        invalid = result[column].notna() & (result[column] <= 1.0)
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 1보다 커야 합니다.")

    for column in ("home_vote_pct", "draw_vote_pct", "away_vote_pct"):
        cleaned = result[column].astype("string").str.replace("%", "", regex=False)
        cleaned = cleaned.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA})
        result[column] = _optional_numeric(cleaned, column)
        invalid = result[column].notna() & (
            (result[column] < 0.0) | (result[column] > 100.0)
        )
        if invalid.any():
            raise ValueError(f"{column} 값은 0~100 범위여야 합니다.")

    for column in ("home_team_id", "away_team_id"):
        result[column] = _optional_integer(result[column], column)

    duplicate_keys = ["product", "round_id", "match_no", "market_raw"]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("같은 회차·경기번호·시장유형 행이 중복되었습니다.")

    ordered = (
        "product",
        "round_id",
        "match_no",
        "sport",
        "league",
        "competition_code",
        "kickoff_kst",
        "deadline_kst",
        "home_team",
        "away_team",
        "market_raw",
        "market_type",
        "home_odds",
        "draw_odds",
        "away_odds",
        "home_vote_pct",
        "draw_vote_pct",
        "away_vote_pct",
        "home_team_id",
        "away_team_id",
        "retrieved_at_kst",
        "source_url",
    )
    return result.loc[:, ordered].sort_values(
        ["round_id", "match_no"], kind="stable", ignore_index=True
    )


def classify_market(value: object) -> str:
    """베트맨 게임유형을 기존 1X2 모델 연결 가능 여부로 분류한다."""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    compact = re.sub(r"\s+", "", text)
    if any(token in compact for token in ("핸디", "handicap")):
        return MARKET_HANDICAP
    if any(token in compact for token in ("언더오버", "u/o", "under", "over")):
        return MARKET_TOTALS
    if compact in {
        "승무패",
        "일반승무패",
        "일반1x2",
        "축구승무패",
        "1x2",
        "일반",
    }:
        return MARKET_1X2
    return MARKET_OTHER


def available_rounds(frame: pd.DataFrame) -> list[str]:
    """파일에 나타난 순서대로 고유 회차를 반환한다."""

    return frame["round_id"].drop_duplicates().astype(str).tolist()


def select_round(frame: pd.DataFrame, round_id: str) -> pd.DataFrame:
    selected = frame.loc[frame["round_id"].astype(str) == str(round_id)].copy()
    if selected.empty:
        raise ValueError(f"{round_id} 회차 데이터가 없습니다.")
    return selected.sort_values("match_no", kind="stable", ignore_index=True)


def validate_toto_round(frame: pd.DataFrame) -> None:
    """축구토토 승무패 회차가 정확히 1~14번 경기인지 확인한다."""

    if len(frame) != 14:
        raise ValueError(
            f"베트맨 승무패 회차에는 정확히 14경기가 필요합니다. 현재 {len(frame)}경기입니다."
        )
    numbers = frame["match_no"].astype(int).tolist()
    if numbers != list(range(1, 15)):
        raise ValueError("경기번호는 공식 순서 1~14가 각각 한 번씩 있어야 합니다.")
    if not frame["sport"].str.contains("축구", case=False, regex=False).all():
        raise ValueError("승무패 14경기 파일에 축구가 아닌 종목이 포함되어 있습니다.")
    if not (frame["market_type"] == MARKET_1X2).all():
        raise ValueError("승무패 14경기는 일반 승무패 시장만 포함해야 합니다.")


def display_betman_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Streamlit 표에 필요한 열만 한국어로 표시한다."""

    output = pd.DataFrame(
        {
            "경기번호": frame["match_no"],
            "경기시각(KST)": frame["kickoff_kst"].map(_format_timestamp),
            "마감시각(KST)": frame["deadline_kst"].map(_format_timestamp),
            "종목": frame["sport"],
            "대회": frame["league"],
            "홈팀": frame["home_team"],
            "원정팀": frame["away_team"],
            "시장 구분": frame["market_type"],
            "홈승 배당": frame["home_odds"],
            "무승부 배당": frame["draw_odds"],
            "원정승 배당": frame["away_odds"],
            "승 투표율(%)": frame["home_vote_pct"],
            "무 투표율(%)": frame["draw_vote_pct"],
            "패 투표율(%)": frame["away_vote_pct"],
            "대회코드": frame["competition_code"],
        }
    )
    return output.reset_index(drop=True)


def latest_retrieved_at(frame: pd.DataFrame) -> str:
    timestamps = frame["retrieved_at_kst"].dropna()
    if timestamps.empty:
        return "파일에 조회 기준시각 없음"
    return _format_timestamp(max(timestamps))


def link_betman_matches(
    betman_frame: pd.DataFrame,
    fixture_frame: pd.DataFrame,
    *,
    alias_frame: pd.DataFrame | None = None,
    supported_codes: Iterable[str] = (),
    tolerance_hours: float = 18.0,
) -> BetmanLinkResult:
    """베트맨 행을 보수적으로 매칭하고 기존 모델 입력을 만든다.

    문자열 유사도는 사용하지 않는다. 팀 ID가 양쪽에 있으면 ID를 우선하며,
    그렇지 않으면 명시적 별칭 또는 정규화된 완전 일치만 허용한다.
    """

    aliases = _build_alias_index(alias_frame)
    supported = {str(code).strip().upper() for code in supported_codes}
    fixtures = fixture_frame.copy()
    if not fixtures.empty:
        _validate_fixture_columns(fixtures)
        fixtures["대회코드"] = fixtures["대회코드"].astype("string").str.upper()
        fixtures["경기시각(KST)"] = fixtures["경기시각(KST)"].map(
            lambda value: _parse_kst_datetime(value, required=True, label="football-data 경기시각")
        )

    model_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for _, row in betman_frame.sort_values("match_no", kind="stable").iterrows():
        code = str(row["competition_code"]).strip().upper()
        base_status = {
            "경기번호": int(row["match_no"]),
            "경기": f"{row['home_team']} vs {row['away_team']}",
            "대회코드": code or "없음",
            "시장 구분": row["market_type"],
        }
        if row["market_type"] != MARKET_1X2:
            status_rows.append(
                {**base_status, "매칭 상태": "분석 불가 시장", "사유": "1X2 모델은 일반 승무패만 지원"}
            )
            continue
        if not code:
            status_rows.append(
                {**base_status, "매칭 상태": "매칭 실패", "사유": "대회코드가 필요함"}
            )
            continue
        if supported and code not in supported:
            status_rows.append(
                {**base_status, "매칭 상태": "전력 데이터 부족", "사유": "무료 API 미지원 대회"}
            )
            continue

        candidates = fixtures.loc[fixtures["대회코드"] == code]
        kickoff = row["kickoff_kst"]
        candidates = candidates.loc[
            candidates["경기시각(KST)"].map(
                lambda value: abs((value - kickoff).total_seconds())
                <= tolerance_hours * 3600
            )
        ]
        matched_candidates = []
        for candidate_index, candidate in candidates.iterrows():
            if not _team_matches(
                row["home_team"],
                candidate["홈팀"],
                code,
                row.get("home_team_id"),
                candidate.get("홈팀 ID"),
                aliases,
            ):
                continue
            if not _team_matches(
                row["away_team"],
                candidate["원정팀"],
                code,
                row.get("away_team_id"),
                candidate.get("원정팀 ID"),
                aliases,
            ):
                continue
            matched_candidates.append((candidate_index, candidate))

        if not matched_candidates:
            status_rows.append(
                {
                    **base_status,
                    "매칭 상태": "매칭 실패",
                    "사유": "날짜·대회·홈/원정팀 완전 일치 후보 없음",
                }
            )
            continue
        if len(matched_candidates) > 1:
            candidate_names = ", ".join(
                f"{candidate['홈팀']} vs {candidate['원정팀']}"
                for _, candidate in matched_candidates
            )
            status_rows.append(
                {
                    **base_status,
                    "매칭 상태": "사용자 확인 필요",
                    "사유": f"중복 후보 {len(matched_candidates)}개: {candidate_names}",
                }
            )
            continue

        _, candidate = matched_candidates[0]
        model_rows.append(
            {
                "경기번호": str(int(row["match_no"])),
                "홈팀": str(row["home_team"]),
                "원정팀": str(row["away_team"]),
                "홈 기대득점": candidate["홈 기대득점"],
                "원정 기대득점": candidate["원정 기대득점"],
                "홈 Elo": candidate["홈 Elo"],
                "원정 Elo": candidate["원정 Elo"],
                "홈승 배당": row["home_odds"],
                "무승부 배당": row["draw_odds"],
                "원정승 배당": row["away_odds"],
                "데이터 구분": "베트맨 사용자 파일 + football-data.org 모델 추정",
            }
        )
        status_rows.append(
            {
                **base_status,
                "매칭 상태": "자동 매칭 완료",
                "사유": f"{candidate['홈팀']} vs {candidate['원정팀']}",
            }
        )

    return BetmanLinkResult(
        model_frame=pd.DataFrame(model_rows),
        status_frame=pd.DataFrame(status_rows),
    )


def _read_table_file(payload: bytes, filename: str, *, label: str) -> pd.DataFrame:
    if not payload:
        raise ValueError(f"{label}이 비어 있습니다.")
    lower_name = filename.lower()
    if lower_name.endswith(".xlsx"):
        try:
            return pd.read_excel(BytesIO(payload), engine="openpyxl")
        except Exception as exc:
            raise ValueError(f"{label}의 XLSX 형식을 읽을 수 없습니다.") from exc
    if not lower_name.endswith(".csv"):
        raise ValueError(f"{label}은 CSV 또는 XLSX만 지원합니다.")
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(BytesIO(payload), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{label} 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949를 사용하세요.")


def _normalise_header(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", "_", text)


def _normalise_team(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def _parse_match_number(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    if match is None:
        return None
    parsed = int(match.group())
    return parsed if parsed >= 1 else None


def _parse_kst_datetime(value: object, *, required: bool, label: str) -> pd.Timestamp:
    if value is None or pd.isna(value) or str(value).strip() == "":
        if required:
            raise ValueError(f"{label}에 빈 값이 있습니다.")
        return pd.NaT
    cleaned = re.sub(r"\([^)]*\)", " ", str(value)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    try:
        parsed = pd.Timestamp(cleaned)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 형식을 읽을 수 없습니다: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.tz_localize(SEOUL_TIMEZONE)
    return parsed.tz_convert(SEOUL_TIMEZONE)


def _optional_numeric(values: pd.Series, label: str) -> pd.Series:
    try:
        return pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 열에는 숫자 또는 빈 값만 입력할 수 있습니다.") from exc


def _optional_integer(values: pd.Series, label: str) -> pd.Series:
    numeric = _optional_numeric(values, label)
    invalid = numeric.notna() & ((numeric % 1 != 0) | (numeric <= 0))
    if invalid.any():
        raise ValueError(f"{label} 값은 비워 두거나 1 이상의 정수여야 합니다.")
    return numeric.astype("Int64")


def _format_timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(SEOUL_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(SEOUL_TIMEZONE)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _build_alias_index(
    alias_frame: pd.DataFrame | None,
) -> dict[tuple[str, str], set[str]]:
    index: dict[tuple[str, str], set[str]] = {}
    if alias_frame is None or alias_frame.empty:
        return index
    required = {"competition_code", "source_team", "target_team"}
    if not required.issubset(alias_frame.columns):
        raise ValueError("팀명 매핑 표준 열이 없습니다.")
    for _, row in alias_frame.iterrows():
        code = str(row["competition_code"]).strip().upper()
        source = _normalise_team(row["source_team"])
        target = _normalise_team(row["target_team"])
        index.setdefault((code, source), set()).add(target)
    ambiguous = [key for key, targets in index.items() if len(targets) > 1]
    if ambiguous:
        raise ValueError(
            "같은 대회·베트맨 팀명에 서로 다른 football-data 팀명이 지정되었습니다."
        )
    return index


def _team_matches(
    source_name: object,
    target_name: object,
    competition_code: str,
    source_id: object,
    target_id: object,
    aliases: dict[tuple[str, str], set[str]],
) -> bool:
    if not pd.isna(source_id) and not pd.isna(target_id):
        return int(source_id) == int(target_id)
    source = _normalise_team(source_name)
    target = _normalise_team(target_name)
    if source == target:
        return True
    mapped = set(aliases.get((competition_code, source), set()))
    mapped.update(aliases.get(("", source), set()))
    return target in mapped


def _validate_fixture_columns(frame: pd.DataFrame) -> None:
    required = {
        "대회코드",
        "경기시각(KST)",
        "홈팀",
        "원정팀",
        "홈 기대득점",
        "원정 기대득점",
        "홈 Elo",
        "원정 Elo",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("football-data 경기 표준 열이 없습니다: " + ", ".join(missing))
