"""과거 회차 예측 스냅샷과 실제 결과 평가를 분리하는 백테스트 모듈.

예측 단계는 :class:`BacktestContext`와 경기 전 모델 입력만 받는다. 실제 스코어는
별도의 :class:`OutcomeSet`에 보관하고 ``evaluate_snapshot`` 단계에서만 결합한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
import math
import re
import unicodedata
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from betman_data import MARKET_1X2, classify_market
from football_data import (
    HistoricalMatch,
    calculate_elo_ratings,
    calculate_team_form,
    estimate_expected_goals,
)
from models import (
    CombinationSearchResult,
    MatchAnalysis,
    ModelConfig,
    OUTCOMES,
    analyze_dataframe,
    calculate_top_combinations,
)


SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")
MODEL_VERSION = "elo-poisson-dixon-coles-v1"
PRODUCT_PROTO = "프로토 승부식"
PRODUCT_TOTO = "축구토토 승무패"
MODE_STORED = "경기 전 입력값으로 현재 재계산"
MODE_RECONSTRUCTED = "현재 확보한 역사적 결과 재구성"

BACKTEST_TEMPLATE_COLUMNS = (
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
    "홈득점",
    "원정득점",
    "홈팀 ID",
    "원정팀 ID",
    "홈 기대득점",
    "원정 기대득점",
    "홈 Elo",
    "원정 Elo",
    "스냅샷시각(KST)",
    "홈승 배당",
    "무승부 배당",
    "원정승 배당",
    "조회기준시각(KST)",
    "출처 URL",
    "데이터 구분",
)

_COLUMN_ALIASES = {
    "상품": "product",
    "product": "product",
    "회차": "round_id",
    "round": "round_id",
    "round_id": "round_id",
    "경기번호": "match_no",
    "match_no": "match_no",
    "match_number": "match_no",
    "종목": "sport",
    "sport": "sport",
    "대회": "league",
    "리그": "league",
    "league": "league",
    "대회코드": "competition_code",
    "competition_code": "competition_code",
    "경기시각(kst)": "kickoff_kst",
    "경기시각": "kickoff_kst",
    "kickoff_kst": "kickoff_kst",
    "마감시각(kst)": "deadline_kst",
    "마감시각": "deadline_kst",
    "deadline_kst": "deadline_kst",
    "홈팀": "home_team",
    "home_team": "home_team",
    "원정팀": "away_team",
    "away_team": "away_team",
    "시장유형": "market_raw",
    "game_type": "market_raw",
    "market": "market_raw",
    "홈득점": "home_goals",
    "home_goals": "home_goals",
    "원정득점": "away_goals",
    "away_goals": "away_goals",
    "홈팀_id": "home_team_id",
    "홈팀id": "home_team_id",
    "home_team_id": "home_team_id",
    "원정팀_id": "away_team_id",
    "원정팀id": "away_team_id",
    "away_team_id": "away_team_id",
    "홈_기대득점": "home_xg",
    "홈기대득점": "home_xg",
    "home_xg": "home_xg",
    "원정_기대득점": "away_xg",
    "원정기대득점": "away_xg",
    "away_xg": "away_xg",
    "홈_elo": "home_elo",
    "홈elo": "home_elo",
    "home_elo": "home_elo",
    "원정_elo": "away_elo",
    "원정elo": "away_elo",
    "away_elo": "away_elo",
    "스냅샷시각(kst)": "snapshot_at_kst",
    "스냅샷시각": "snapshot_at_kst",
    "snapshot_at_kst": "snapshot_at_kst",
    "홈승_배당": "home_odds",
    "홈승배당": "home_odds",
    "home_odds": "home_odds",
    "무승부_배당": "draw_odds",
    "무승부배당": "draw_odds",
    "draw_odds": "draw_odds",
    "원정승_배당": "away_odds",
    "원정승배당": "away_odds",
    "away_odds": "away_odds",
    "조회기준시각(kst)": "retrieved_at_kst",
    "조회기준시각": "retrieved_at_kst",
    "retrieved_at_kst": "retrieved_at_kst",
    "출처_url": "source_url",
    "출처url": "source_url",
    "source_url": "source_url",
    "데이터_구분": "data_type",
    "데이터구분": "data_type",
    "data_type": "data_type",
}

_REQUIRED_COLUMNS = (
    "product",
    "round_id",
    "match_no",
    "sport",
    "league",
    "competition_code",
    "kickoff_kst",
    "home_team",
    "away_team",
    "market_raw",
)


@dataclass(frozen=True)
class BacktestContext:
    product: str
    round_id: str
    cutoff: pd.Timestamp
    cutoff_basis: str
    target_frame: pd.DataFrame
    source_label: str
    acquired_at_label: str


@dataclass(frozen=True)
class OutcomeSet:
    product: str
    round_id: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class PreparedRound:
    context: BacktestContext
    outcomes: OutcomeSet


@dataclass(frozen=True)
class BacktestModelInput:
    frame: pd.DataFrame
    evidence_frame: pd.DataFrame
    mode: str
    history_match_count: int
    history_start: str
    history_end: str
    history_sources: str = "기록 없음"


@dataclass(frozen=True)
class PredictionSnapshot:
    snapshot_id: str
    product: str
    round_id: str
    cutoff: pd.Timestamp
    cutoff_basis: str
    model_version: str
    mode: str
    source_label: str
    acquired_at_label: str
    created_at: pd.Timestamp
    history_match_count: int
    history_start: str
    history_end: str
    predictions: pd.DataFrame
    top_combinations: CombinationSearchResult | None
    model_config: dict[str, float | int] | None = None
    history_sources: str = "기록 없음"


@dataclass(frozen=True)
class BacktestEvaluation:
    snapshot_id: str
    product: str
    round_id: str
    total_matches: int
    correct_matches: int
    accuracy: float
    brier_score: float
    log_loss: float
    match_frame: pd.DataFrame
    league_frame: pd.DataFrame
    trend_frame: pd.DataFrame
    actual_combination: str | None
    actual_in_top20: bool | None
    actual_rank: int | None
    top1_matching_matches: int | None
    all_14_correct: bool | None


@dataclass(frozen=True)
class AggregateEvaluation:
    total_matches: int
    correct_matches: int
    accuracy: float
    brier_score: float
    log_loss: float
    league_frame: pd.DataFrame
    trend_frame: pd.DataFrame
    match_frame: pd.DataFrame


def backtest_template_csv() -> bytes:
    """실제 경기값을 포함하지 않는 빈 업로드 템플릿."""

    return (
        pd.DataFrame(columns=BACKTEST_TEMPLATE_COLUMNS)
        .to_csv(index=False)
        .encode("utf-8-sig")
    )


def history_template_csv() -> bytes:
    return pd.DataFrame(columns=(
        "대회코드", "경기시작(KST)", "결과확정시각(KST)", "홈팀 ID", "홈팀",
        "원정팀 ID", "원정팀", "홈득점", "원정득점", "출처", "확보시각(KST)",
    )).to_csv(index=False).encode("utf-8-sig")


def read_history_file(payload: bytes, filename: str) -> dict[str, tuple[HistoricalMatch, ...]]:
    """출처와 결과 확정시각이 명시된 사용자의 이력 파일을 읽는다."""
    if not payload:
        raise ValueError("과거 이력 파일이 비어 있습니다.")
    if filename.lower().endswith(".xlsx"):
        try:
            frame = pd.read_excel(BytesIO(payload), engine="openpyxl")
        except Exception as exc:
            raise ValueError("과거 이력 XLSX를 읽을 수 없습니다.") from exc
    elif filename.lower().endswith(".csv"):
        frame = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                frame = pd.read_csv(BytesIO(payload), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError("이력 CSV 인코딩을 읽을 수 없습니다.")
    else:
        raise ValueError("이력 파일은 CSV 또는 XLSX만 지원합니다.")
    required = ("대회코드", "경기시작(KST)", "결과확정시각(KST)", "홈팀 ID", "홈팀",
                "원정팀 ID", "원정팀", "홈득점", "원정득점", "출처", "확보시각(KST)")
    missing = [column for column in required if column not in frame.columns]
    if missing or frame.empty:
        raise ValueError("이력 파일이 비었거나 필수 열이 없습니다: " + ", ".join(missing))
    groups: dict[str, list[HistoricalMatch]] = {}
    for index, row in frame.iterrows():
        if any(pd.isna(row[column]) or str(row[column]).strip() == "" for column in required):
            raise ValueError(f"이력 {index + 1}행에 필수 정보가 없습니다.")
        kickoff = _parse_kst(row["경기시작(KST)"], required=True, label="경기시작")
        finished = _parse_kst(row["결과확정시각(KST)"], required=True, label="결과확정시각")
        acquired = _parse_kst(row["확보시각(KST)"], required=True, label="확보시각")
        if finished < kickoff or acquired < finished:
            raise ValueError(f"이력 {index + 1}행의 시작·결과확정·확보시각 순서가 잘못됐습니다.")
        try:
            numbers = [float(row[column]) for column in ("홈팀 ID", "원정팀 ID", "홈득점", "원정득점")]
            if not all(math.isfinite(value) and value.is_integer() for value in numbers):
                raise ValueError
            home_id, away_id, home_goals, away_goals = map(int, numbers)
            if min(home_id, away_id) < 1 or min(home_goals, away_goals) < 0 or home_id == away_id:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"이력 {index + 1}행의 팀 ID 또는 점수가 올바르지 않습니다.") from exc
        code = str(row["대회코드"]).strip().upper()
        groups.setdefault(code, []).append(HistoricalMatch(
            match_id=f"{code}-{index + 1}",
            kickoff=kickoff.to_pydatetime(),
            finished_at=finished.to_pydatetime(),
            home_team_id=home_id,
            home_team=str(row["홈팀"]).strip(),
            away_team_id=away_id,
            away_team=str(row["원정팀"]).strip(),
            home_goals=home_goals,
            away_goals=away_goals,
            source=str(row["출처"]).strip(),
            acquired_at=acquired.to_pydatetime(),
        ))
    return {code: tuple(matches) for code, matches in groups.items()}


def read_backtest_file(payload: bytes, filename: str) -> pd.DataFrame:
    if not payload:
        raise ValueError("과거 회차 파일이 비어 있습니다.")
    lower = filename.lower()
    if lower.endswith(".xlsx"):
        try:
            frame = pd.read_excel(BytesIO(payload), engine="openpyxl")
        except Exception as exc:
            raise ValueError("과거 회차 XLSX를 읽을 수 없습니다.") from exc
    elif lower.endswith(".csv"):
        frame = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                frame = pd.read_csv(BytesIO(payload), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949를 사용하세요.")
    else:
        raise ValueError("과거 회차 파일은 CSV 또는 XLSX만 지원합니다.")
    return normalise_backtest_dataframe(frame)


def normalise_backtest_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """과거 회차 파일을 정규화하되 결과 누락 행은 상태 표시에 남겨 둔다."""

    if frame.empty:
        raise ValueError("과거 회차 파일에 데이터 행이 없습니다.")
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
        raise ValueError("백테스트 필수 열이 없습니다: " + ", ".join(missing))

    defaults: Mapping[str, object] = {
        "deadline_kst": pd.NaT,
        "home_goals": math.nan,
        "away_goals": math.nan,
        "home_team_id": pd.NA,
        "away_team_id": pd.NA,
        "home_xg": math.nan,
        "away_xg": math.nan,
        "home_elo": math.nan,
        "away_elo": math.nan,
        "snapshot_at_kst": pd.NaT,
        "home_odds": math.nan,
        "draw_odds": math.nan,
        "away_odds": math.nan,
        "retrieved_at_kst": pd.NaT,
        "source_url": "",
        "data_type": "사용자 제공 과거 회차",
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
        "data_type",
    ):
        result[column] = result[column].astype("string").fillna("").str.strip()
    for column in _REQUIRED_COLUMNS:
        if column == "match_no":
            continue
        if column != "kickoff_kst" and (result[column] == "").any():
            raise ValueError(f"{column} 열에 빈 값이 있습니다.")

    result["match_no"] = result["match_no"].map(_parse_positive_integer)
    if result["match_no"].isna().any():
        raise ValueError("경기번호는 1 이상의 숫자여야 합니다.")
    result["match_no"] = result["match_no"].astype(int)
    result["kickoff_kst"] = result["kickoff_kst"].map(
        lambda value: _parse_kst(value, required=True, label="경기시각")
    )
    for column, label in (
        ("deadline_kst", "마감시각"),
        ("snapshot_at_kst", "스냅샷시각"),
        ("retrieved_at_kst", "조회기준시각"),
    ):
        result[column] = result[column].map(
            lambda value, item_label=label: _parse_kst(
                value, required=False, label=item_label
            )
        )
    result["market_type"] = result["market_raw"].map(classify_market)

    for column in ("home_goals", "away_goals"):
        result[column] = _optional_numeric(result[column], column)
        invalid = result[column].notna() & (
            (result[column] < 0) | (result[column] % 1 != 0)
        )
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 0 이상의 정수여야 합니다.")
        result[column] = result[column].astype("Int64")
    for column in ("home_team_id", "away_team_id"):
        result[column] = _optional_numeric(result[column], column)
        invalid = result[column].notna() & (
            (result[column] <= 0) | (result[column] % 1 != 0)
        )
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 1 이상의 정수여야 합니다.")
        result[column] = result[column].astype("Int64")
    for column in (
        "home_xg",
        "away_xg",
        "home_elo",
        "away_elo",
        "home_odds",
        "draw_odds",
        "away_odds",
    ):
        result[column] = _optional_numeric(result[column], column)

    for column in ("home_xg", "away_xg"):
        invalid = result[column].notna() & (
            (result[column] <= 0) | (result[column] > 10)
        )
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 0보다 크고 10 이하여야 합니다.")
    for column in ("home_elo", "away_elo"):
        invalid = result[column].notna() & (
            (result[column] < 800) | (result[column] > 2800)
        )
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 800~2800 범위여야 합니다.")
    for column in ("home_odds", "draw_odds", "away_odds"):
        invalid = result[column].notna() & (result[column] <= 1)
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 1보다 커야 합니다.")

    duplicate_keys = ["product", "round_id", "match_no", "market_raw"]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("같은 상품·회차·경기번호·시장 행이 중복되었습니다.")

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
        "home_goals",
        "away_goals",
        "home_team_id",
        "away_team_id",
        "home_xg",
        "away_xg",
        "home_elo",
        "away_elo",
        "snapshot_at_kst",
        "home_odds",
        "draw_odds",
        "away_odds",
        "retrieved_at_kst",
        "source_url",
        "data_type",
    )
    return result.loc[:, ordered].sort_values(
        ["product", "round_id", "match_no"], kind="stable", ignore_index=True
    )


def product_rows(frame: pd.DataFrame, product: str) -> pd.DataFrame:
    if product == PRODUCT_TOTO:
        mask = frame["product"].str.contains("승무패", regex=False)
    elif product == PRODUCT_PROTO:
        mask = frame["product"].str.contains("프로토", regex=False)
    else:
        raise ValueError(f"지원하지 않는 상품입니다: {product}")
    return frame.loc[mask].copy()


def round_status_frame(frame: pd.DataFrame, product: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = product_rows(frame, product)
    for round_id, group in selected.groupby("round_id", sort=False):
        eligible = group.loc[
            group["sport"].str.contains("축구", case=False, regex=False)
            & (group["market_type"] == MARKET_1X2)
        ]
        missing_results = int(
            (eligible["home_goals"].isna() | eligible["away_goals"].isna()).sum()
        )
        expected_count = 14 if product == PRODUCT_TOTO else len(eligible)
        complete = bool(
            len(eligible) > 0
            and missing_results == 0
            and not eligible["match_no"].duplicated().any()
            and (product != PRODUCT_TOTO or
                 sorted(eligible["match_no"].tolist()) == list(range(1, 15)))
        )
        rows.append(
            {
                "회차": str(round_id),
                "분석 대상 경기": len(eligible),
                "결과 누락": missing_results,
                "필요 경기": expected_count,
                "상태": "완료" if complete else "불완전",
            }
        )
    return pd.DataFrame(rows)


def prepare_round(frame: pd.DataFrame, product: str, round_id: str) -> PreparedRound:
    """한 회차를 결과 없는 예측 문맥과 별도 정답 집합으로 즉시 분리한다."""

    selected = product_rows(frame, product)
    selected = selected.loc[selected["round_id"].astype(str) == str(round_id)].copy()
    selected = selected.loc[
        selected["sport"].str.contains("축구", case=False, regex=False)
        & (selected["market_type"] == MARKET_1X2)
    ].sort_values("match_no", kind="stable", ignore_index=True)
    if selected.empty:
        raise ValueError("선택 회차에 축구 일반 승무패 경기가 없습니다.")
    if selected["home_goals"].isna().any() or selected["away_goals"].isna().any():
        missing = selected.loc[
            selected["home_goals"].isna() | selected["away_goals"].isna(), "match_no"
        ].tolist()
        raise ValueError(f"실제 결과가 없는 경기번호: {missing}")
    if product == PRODUCT_TOTO:
        numbers = selected["match_no"].astype(int).tolist()
        if len(selected) != 14 or numbers != list(range(1, 15)):
            raise ValueError("승무패 백테스트는 공식 순서 1~14의 정확히 14경기가 필요합니다.")

    deadlines = selected["deadline_kst"].dropna().sort_values()
    if len(deadlines) != len(selected):
        cutoff = pd.Timestamp(selected["kickoff_kst"].min().to_pydatetime() - timedelta(minutes=1))
        if not deadlines.empty:
            cutoff = min(cutoff, deadlines.iloc[0])
        cutoff_basis = "마감시각 없음 · 첫 경기 시작 1분 전(보수적 대체)"
    else:
        cutoff = deadlines.iloc[0]
        cutoff_basis = (
            "회차 공통 마감시각"
            if deadlines.nunique() == 1
            else "파일 내 가장 이른 마감시각(보수적)"
        )
    if cutoff >= selected["kickoff_kst"].min():
        raise ValueError("분석 기준시각은 모든 대상 경기 시작 전이어야 합니다.")
    if selected["match_no"].duplicated().any():
        raise ValueError("동일 회차에 일반 승무패 경기번호가 중복되었습니다.")

    target_columns = (
        "match_no",
        "league",
        "competition_code",
        "kickoff_kst",
        "deadline_kst",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "home_xg",
        "away_xg",
        "home_elo",
        "away_elo",
        "snapshot_at_kst",
        "home_odds",
        "draw_odds",
        "away_odds",
        "data_type",
    )
    targets = selected.loc[:, target_columns].copy()
    outcomes = selected.loc[
        :, ("match_no", "home_team", "away_team", "home_goals", "away_goals")
    ].copy()
    source_values = [value for value in selected["source_url"].unique() if value]
    source_label = ", ".join(source_values) if source_values else "파일에 출처 없음"
    acquired = selected["retrieved_at_kst"].dropna()
    acquired_label = (
        _format_timestamp(acquired.max()) if not acquired.empty else "파일에 확보시각 없음"
    )
    return PreparedRound(
        context=BacktestContext(
            product=product,
            round_id=str(round_id),
            cutoff=cutoff,
            cutoff_basis=cutoff_basis,
            target_frame=targets,
            source_label=source_label,
            acquired_at_label=acquired_label,
        ),
        outcomes=OutcomeSet(product=product, round_id=str(round_id), frame=outcomes),
    )


def snapshot_model_input(context: BacktestContext) -> BacktestModelInput:
    """파일에 기록된 경기 전 특징만 모델 입력으로 변환한다."""

    frame = context.target_frame
    required = ("home_xg", "away_xg", "home_elo", "away_elo", "snapshot_at_kst")
    missing_rows = frame.loc[:, required].isna().any(axis=1)
    if missing_rows.any():
        numbers = frame.loc[missing_rows, "match_no"].tolist()
        raise ValueError(f"경기 전 스냅샷 특징이 누락된 경기번호: {numbers}")
    invalid_time = frame["snapshot_at_kst"] > context.cutoff
    if invalid_time.any():
        numbers = frame.loc[invalid_time, "match_no"].tolist()
        raise ValueError(f"분석 기준시각 이후 스냅샷은 사용할 수 없습니다: {numbers}")

    model_frame = _standard_model_frame(frame, data_type=MODE_STORED)
    evidence = pd.DataFrame(
        {
            "경기번호": frame["match_no"],
            "스냅샷시각(KST)": frame["snapshot_at_kst"].map(_format_timestamp),
            "분석 기준시각(KST)": _format_timestamp(context.cutoff),
            "입력 구분": MODE_STORED,
        }
    )
    snapshot_times = frame["snapshot_at_kst"].dropna()
    return BacktestModelInput(
        frame=model_frame,
        evidence_frame=evidence,
        mode=MODE_STORED,
        history_match_count=0,
        history_start=(
            _format_timestamp(snapshot_times.min()) if not snapshot_times.empty else "기록 없음"
        ),
        history_end=(
            _format_timestamp(snapshot_times.max()) if not snapshot_times.empty else "기록 없음"
        ),
    )


def reconstructed_model_input(
    context: BacktestContext,
    history_by_competition: Mapping[str, Sequence[HistoricalMatch]],
) -> BacktestModelInput:
    """기준시각 이전 완료 경기만으로 Elo·최근 폼·기대득점을 재계산한다."""

    cutoff_utc = context.cutoff.tz_convert(timezone.utc).to_pydatetime()
    if context.cutoff >= context.target_frame["kickoff_kst"].min():
        raise ValueError("분석 기준시각은 모든 대상 경기 시작 전이어야 합니다.")
    model_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    used_matches: dict[str, HistoricalMatch] = {}
    for _, target in context.target_frame.sort_values("match_no", kind="stable").iterrows():
        code = str(target["competition_code"]).strip().upper()
        available = list(history_by_competition.get(code, ()))
        prior = sorted(
            (match for match in available
             if match.finished_at is not None
             and match.kickoff <= match.finished_at < cutoff_utc),
            key=lambda match: match.kickoff,
        )
        for match in prior:
            used_matches[match.match_id] = match
        home_id = _resolve_team_id(target, prior, side="home")
        away_id = _resolve_team_id(target, prior, side="away")
        ratings = calculate_elo_ratings(prior)
        home_form = calculate_team_form(
            prior,
            team_id=home_id,
            team_name=str(target["home_team"]),
            elo=ratings.get(home_id, 1500.0),
            prediction_venue="home",
        )
        away_form = calculate_team_form(
            prior,
            team_id=away_id,
            team_name=str(target["away_team"]),
            elo=ratings.get(away_id, 1500.0),
            prediction_venue="away",
        )
        home_xg, away_xg = estimate_expected_goals(
            prior, home_id=home_id, away_id=away_id
        )
        model_rows.append(
            {
                "경기번호": str(int(target["match_no"])),
                "홈팀": str(target["home_team"]),
                "원정팀": str(target["away_team"]),
                "홈 기대득점": round(home_xg, 3),
                "원정 기대득점": round(away_xg, 3),
                "홈 Elo": round(home_form.elo, 1),
                "원정 Elo": round(away_form.elo, 1),
                "홈승 배당": target["home_odds"],
                "무승부 배당": target["draw_odds"],
                "원정승 배당": target["away_odds"],
                "데이터 구분": MODE_RECONSTRUCTED,
            }
        )
        evidence_rows.append(
            {
                "경기번호": int(target["match_no"]),
                "대회코드": code,
                "기준시각 이전 완료 경기": len(prior),
                "홈 최근5": home_form.recent5_record,
                "원정 최근5": away_form.recent5_record,
                "홈 최근10": home_form.recent10_record,
                "원정 최근10": away_form.recent10_record,
                "홈 최근10 득점 평균": home_form.recent10_goals_for,
                "홈 최근10 실점 평균": home_form.recent10_goals_against,
                "원정 최근10 득점 평균": away_form.recent10_goals_for,
                "원정 최근10 실점 평균": away_form.recent10_goals_against,
                "홈 Elo": round(home_form.elo, 1),
                "원정 Elo": round(away_form.elo, 1),
                "추정 기대득점": f"{home_xg:.2f} - {away_xg:.2f}",
                "불확실성": max(
                    (home_form.uncertainty, away_form.uncertainty),
                    key={"낮음": 0, "보통": 1, "높음": 2}.get,
                ),
            }
        )
    history = sorted(used_matches.values(), key=lambda match: match.kickoff)
    return BacktestModelInput(
        frame=pd.DataFrame(model_rows),
        evidence_frame=pd.DataFrame(evidence_rows),
        mode=MODE_RECONSTRUCTED,
        history_match_count=len(history),
        history_start=_history_timestamp(history, first=True),
        history_end=_history_timestamp(history, first=False),
        history_sources="; ".join(sorted({
            f"{match.source} (확보: {_format_timestamp(match.acquired_at)})"
            for match in history if match.source and match.acquired_at is not None
        })) or "기록 없음",
    )


def create_prediction_snapshot(
    context: BacktestContext,
    model_input: BacktestModelInput,
    config: ModelConfig | None = None,
    *,
    created_at: datetime | pd.Timestamp | None = None,
) -> PredictionSnapshot:
    """실제 결과를 인자로 받지 않고 불변 식별자가 있는 예측 스냅샷을 만든다."""

    config = config or ModelConfig()
    _, _, analyses = analyze_dataframe(
        model_input.frame,
        config,
        default_data_type=model_input.mode,
    )
    target = context.target_frame.set_index("match_no", drop=False)
    rows: list[dict[str, object]] = []
    for analysis in analyses:
        match_no = int(analysis.match_id)
        metadata = target.loc[match_no]
        probabilities = analysis.probabilities
        predicted_index = int(np.argmax(probabilities))
        rows.append(
            {
                "round_id": context.round_id,
                "product": context.product,
                "match_no": match_no,
                "match_date": metadata["kickoff_kst"].date().isoformat(),
                "kickoff_kst": _format_timestamp(metadata["kickoff_kst"]),
                "league": str(metadata["league"]),
                "home_team": analysis.home_team,
                "away_team": analysis.away_team,
                "home_probability": analysis.home_win,
                "draw_probability": analysis.draw,
                "away_probability": analysis.away_win,
                "predicted_outcome": OUTCOMES[predicted_index],
                "predicted_score": (
                    f"{analysis.predicted_home_goals}:{analysis.predicted_away_goals}"
                ),
            }
        )
    predictions = pd.DataFrame(rows).sort_values(
        "match_no", kind="stable", ignore_index=True
    )
    top_combinations = None
    if context.product == PRODUCT_TOTO:
        if len(analyses) != 14:
            raise ValueError("승무패 예측 스냅샷에는 정확히 14경기가 필요합니다.")
        top_combinations = calculate_top_combinations(analyses, top_n=20)
    timestamp = pd.Timestamp(created_at or datetime.now(timezone.utc))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    fingerprint = {
        "round": context.round_id,
        "product": context.product,
        "cutoff": context.cutoff.isoformat(),
        "model": MODEL_VERSION,
        "mode": model_input.mode,
        "rho": config.dixon_coles_rho,
        "home_advantage": config.home_advantage_elo,
        "max_goals": config.max_goals,
        "elo_goal_scale": config.elo_goal_scale,
        "probabilities": predictions.loc[
            :, ("match_no", "home_probability", "draw_probability", "away_probability")
        ].to_dict(orient="records"),
    }
    snapshot_id = hashlib.sha256(
        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return PredictionSnapshot(
        snapshot_id=snapshot_id,
        product=context.product,
        round_id=context.round_id,
        cutoff=context.cutoff,
        cutoff_basis=context.cutoff_basis,
        model_version=MODEL_VERSION,
        mode=model_input.mode,
        source_label=context.source_label,
        acquired_at_label=context.acquired_at_label,
        created_at=timestamp,
        history_match_count=model_input.history_match_count,
        history_start=model_input.history_start,
        history_end=model_input.history_end,
        predictions=predictions,
        top_combinations=top_combinations,
        model_config={"max_goals": config.max_goals, "dixon_coles_rho": config.dixon_coles_rho,
                      "home_advantage_elo": config.home_advantage_elo, "elo_goal_scale": config.elo_goal_scale},
        history_sources=model_input.history_sources,
    )


def evaluate_snapshot(
    snapshot: PredictionSnapshot, outcomes: OutcomeSet
) -> BacktestEvaluation:
    """저장된 예측을 변경하지 않고 평가 단계에서만 실제 결과를 결합한다."""

    if (snapshot.product, snapshot.round_id) != (outcomes.product, outcomes.round_id):
        raise ValueError("예측 스냅샷과 실제 결과의 상품·회차가 다릅니다.")
    merged = snapshot.predictions.merge(
        outcomes.frame,
        on=["match_no", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(snapshot.predictions) or len(merged) != len(outcomes.frame):
        raise ValueError("예측과 실제 결과의 경기번호·팀 순서가 일치하지 않습니다.")
    merged["actual_outcome"] = merged.apply(
        lambda row: outcome_from_score(int(row["home_goals"]), int(row["away_goals"])),
        axis=1,
    )
    outcome_index = {label: index for index, label in enumerate(OUTCOMES)}
    probability_columns = (
        "home_probability",
        "draw_probability",
        "away_probability",
    )
    brier_values: list[float] = []
    log_values: list[float] = []
    for _, row in merged.iterrows():
        actual_index = outcome_index[str(row["actual_outcome"])]
        probabilities = np.asarray([row[column] for column in probability_columns], dtype=float)
        target = np.zeros(3, dtype=float)
        target[actual_index] = 1.0
        brier_values.append(float(np.square(probabilities - target).sum()))
        log_values.append(float(-math.log(max(probabilities[actual_index], 1e-15))))
    merged["hit"] = merged["predicted_outcome"] == merged["actual_outcome"]
    merged["brier_score"] = brier_values
    merged["log_loss"] = log_values
    merged["actual_score"] = (
        merged["home_goals"].astype(str) + ":" + merged["away_goals"].astype(str)
    )
    total = len(merged)
    correct = int(merged["hit"].sum())
    league_frame = _group_performance(merged, ["league"])
    trend_frame = _group_performance(merged, ["match_date", "round_id"])

    actual_combination = None
    actual_in_top20 = None
    actual_rank = None
    top1_matching = None
    all_14_correct = None
    if snapshot.product == PRODUCT_TOTO:
        if total != 14:
            raise ValueError("승무패 평가에는 정확히 14경기가 필요합니다.")
        actual_labels = tuple(merged.sort_values("match_no")["actual_outcome"].tolist())
        actual_combination = "-".join(actual_labels)
        combinations = snapshot.top_combinations.combinations if snapshot.top_combinations else ()
        for rank, (labels, _) in enumerate(combinations, start=1):
            if labels == actual_labels:
                actual_rank = rank
                break
        actual_in_top20 = actual_rank is not None
        top1 = combinations[0][0] if combinations else tuple()
        top1_matching = sum(
            predicted == actual for predicted, actual in zip(top1, actual_labels, strict=True)
        ) if top1 else 0
        all_14_correct = top1_matching == 14

    display = merged.copy()
    display["예측 스냅샷 ID"] = snapshot.snapshot_id
    display["분석 기준시각(KST)"] = _format_timestamp(snapshot.cutoff)
    display["모델 버전"] = snapshot.model_version
    display["예측 방식"] = snapshot.mode
    display["home_probability"] = display["home_probability"] * 100.0
    display["draw_probability"] = display["draw_probability"] * 100.0
    display["away_probability"] = display["away_probability"] * 100.0
    display = display.rename(
        columns={
            "round_id": "회차",
            "match_no": "경기번호",
            "match_date": "경기일",
            "kickoff_kst": "경기시각(KST)",
            "league": "리그",
            "home_team": "홈팀",
            "away_team": "원정팀",
            "home_probability": "홈승 확률(%)",
            "draw_probability": "무승부 확률(%)",
            "away_probability": "원정승 확률(%)",
            "predicted_outcome": "예측",
            "actual_outcome": "실제",
            "actual_score": "실제 스코어",
            "hit": "적중",
            "brier_score": "Brier Score",
            "log_loss": "Log Loss",
        }
    )
    return BacktestEvaluation(
        snapshot_id=snapshot.snapshot_id,
        product=snapshot.product,
        round_id=snapshot.round_id,
        total_matches=total,
        correct_matches=correct,
        accuracy=correct / total,
        brier_score=float(np.mean(brier_values)),
        log_loss=float(np.mean(log_values)),
        match_frame=display,
        league_frame=league_frame,
        trend_frame=trend_frame,
        actual_combination=actual_combination,
        actual_in_top20=actual_in_top20,
        actual_rank=actual_rank,
        top1_matching_matches=top1_matching,
        all_14_correct=all_14_correct,
    )


def aggregate_evaluations(
    evaluations: Sequence[BacktestEvaluation],
) -> AggregateEvaluation:
    if not evaluations:
        raise ValueError("집계할 백테스트 결과가 없습니다.")
    frames = []
    for evaluation in evaluations:
        frame = evaluation.match_frame.rename(
            columns={
                "회차": "round_id",
                "경기일": "match_date",
                "리그": "league",
                "적중": "hit",
                "Brier Score": "brier_score",
                "Log Loss": "log_loss",
            }
        ).copy()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    total = len(combined)
    correct = int(combined["hit"].sum())
    return AggregateEvaluation(
        total_matches=total,
        correct_matches=correct,
        accuracy=correct / total,
        brier_score=float(combined["brier_score"].mean()),
        log_loss=float(combined["log_loss"].mean()),
        league_frame=_group_performance(combined, ["league"]),
        trend_frame=_group_performance(combined, ["match_date", "round_id"]),
        match_frame=pd.concat([item.match_frame for item in evaluations], ignore_index=True),
    )


def run_snapshot_batch(
    frame: pd.DataFrame,
    product: str,
    round_ids: Iterable[str],
    config: ModelConfig | None = None,
) -> tuple[BacktestEvaluation, ...]:
    """각 회차 예측을 먼저 고정한 다음 그 회차 결과를 평가하는 일괄 경로."""

    prepared_snapshots: list[tuple[PredictionSnapshot, OutcomeSet]] = []
    for round_id in round_ids:
        prepared = prepare_round(frame, product, str(round_id))
        model_input = snapshot_model_input(prepared.context)
        snapshot = create_prediction_snapshot(prepared.context, model_input, config)
        prepared_snapshots.append((snapshot, prepared.outcomes))
    return tuple(evaluate_snapshot(snapshot, outcomes) for snapshot, outcomes in prepared_snapshots)


def prediction_display_frame(snapshot: PredictionSnapshot) -> pd.DataFrame:
    frame = snapshot.predictions.copy()
    for column in ("home_probability", "draw_probability", "away_probability"):
        frame[column] = frame[column] * 100.0
    return frame.rename(
        columns={
            "match_no": "경기번호",
            "kickoff_kst": "경기시각(KST)",
            "league": "리그",
            "home_team": "홈팀",
            "away_team": "원정팀",
            "home_probability": "홈승 확률(%)",
            "draw_probability": "무승부 확률(%)",
            "away_probability": "원정승 확률(%)",
            "predicted_outcome": "예측",
            "predicted_score": "예상 스코어",
        }
    ).loc[
        :,
        (
            "경기번호",
            "경기시각(KST)",
            "리그",
            "홈팀",
            "원정팀",
            "홈승 확률(%)",
            "무승부 확률(%)",
            "원정승 확률(%)",
            "예측",
            "예상 스코어",
        ),
    ]


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "승"
    if home_goals == away_goals:
        return "무"
    return "패"


def _standard_model_frame(frame: pd.DataFrame, *, data_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "경기번호": frame["match_no"].astype(str),
            "홈팀": frame["home_team"],
            "원정팀": frame["away_team"],
            "홈 기대득점": frame["home_xg"],
            "원정 기대득점": frame["away_xg"],
            "홈 Elo": frame["home_elo"],
            "원정 Elo": frame["away_elo"],
            "홈승 배당": frame["home_odds"],
            "무승부 배당": frame["draw_odds"],
            "원정승 배당": frame["away_odds"],
            "데이터 구분": data_type,
        }
    )


def _resolve_team_id(
    target: pd.Series, history: Sequence[HistoricalMatch], *, side: str
) -> int:
    id_column = f"{side}_team_id"
    name_column = f"{side}_team"
    supplied = target[id_column]
    if not pd.isna(supplied):
        team_id = int(supplied)
        known_ids = {
            match.home_team_id
            for match in history
        } | {match.away_team_id for match in history}
        if team_id not in known_ids:
            raise ValueError(
                f"{int(target['match_no'])}번 {target[name_column]}: 기준시각 이전 팀 ID 기록이 없습니다."
            )
        return team_id
    wanted = _normalise_team(target[name_column])
    candidates: set[int] = set()
    for match in history:
        if _normalise_team(match.home_team) == wanted:
            candidates.add(match.home_team_id)
        if _normalise_team(match.away_team) == wanted:
            candidates.add(match.away_team_id)
    if len(candidates) != 1:
        reason = "없음" if not candidates else f"{len(candidates)}개"
        raise ValueError(
            f"{int(target['match_no'])}번 {target[name_column]}: 완전 일치 팀 ID 후보 {reason}."
        )
    return next(iter(candidates))


def _group_performance(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False, sort=False).agg(
        분석경기=("hit", "size"),
        적중경기=("hit", "sum"),
        적중률=("hit", "mean"),
        Brier_Score=("brier_score", "mean"),
        Log_Loss=("log_loss", "mean"),
    )
    output = grouped.reset_index()
    output["적중률(%)"] = output.pop("적중률") * 100.0
    return output.rename(
        columns={
            "league": "리그",
            "match_date": "날짜",
            "round_id": "회차",
            "Brier_Score": "Brier Score",
            "Log_Loss": "Log Loss",
        }
    )


def _normalise_header(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", "_", text)


def _normalise_team(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def _parse_positive_integer(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    if match is None:
        return None
    parsed = int(match.group())
    return parsed if parsed > 0 else None


def _parse_kst(value: object, *, required: bool, label: str) -> pd.Timestamp:
    if value is None or pd.isna(value) or str(value).strip() == "":
        if required:
            raise ValueError(f"{label}에 빈 값이 있습니다.")
        return pd.NaT
    try:
        timestamp = pd.Timestamp(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 형식을 읽을 수 없습니다: {value}") from exc
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(SEOUL_TIMEZONE)
    return timestamp.tz_convert(SEOUL_TIMEZONE)


def _optional_numeric(values: pd.Series, label: str) -> pd.Series:
    cleaned = values.astype("string").replace(
        {"": pd.NA, "<NA>": pd.NA, "nan": pd.NA}
    )
    try:
        numeric = pd.to_numeric(cleaned, errors="raise")
        if not np.isfinite(numeric.dropna().astype(float)).all():
            raise ValueError(f"{label} 열에는 유한한 숫자 또는 빈 값만 입력할 수 있습니다.")
        return numeric
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 열에는 숫자 또는 빈 값만 입력할 수 있습니다.") from exc


def _format_timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(SEOUL_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(SEOUL_TIMEZONE)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _history_timestamp(history: Sequence[HistoricalMatch], *, first: bool) -> str:
    if not history:
        return "기준시각 이전 완료 경기 없음"
    match = history[0] if first else history[-1]
    return pd.Timestamp(match.kickoff).tz_convert(SEOUL_TIMEZONE).strftime("%Y-%m-%d %H:%M")
