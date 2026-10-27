"""Football AI Lab의 확률 모델과 14경기 조합 계산 모듈.

이 모듈은 외부 API에 의존하지 않는다. 입력 기대득점(xG 유사 입력값), Elo와
배당을 이용해 설명 가능한 데모 분석을 제공하며 실제 경기 예측 정확도를
보장하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import heapq
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


OUTCOMES = ("승", "무", "패")


@dataclass(frozen=True)
class ModelConfig:
    """분석 모델 설정값."""

    max_goals: int = 10
    dixon_coles_rho: float = -0.08
    home_advantage_elo: float = 65.0
    elo_goal_scale: float = 0.20

    def __post_init__(self) -> None:
        if self.max_goals < 5:
            raise ValueError("max_goals는 5 이상이어야 합니다.")
        if not -0.2 <= self.dixon_coles_rho <= 0.2:
            raise ValueError("Dixon-Coles rho는 -0.2~0.2 범위여야 합니다.")
        if self.elo_goal_scale < 0:
            raise ValueError("Elo 보정 강도는 0 이상이어야 합니다.")


@dataclass(frozen=True)
class MatchAnalysis:
    match_id: str
    home_team: str
    away_team: str
    adjusted_home_goals: float
    adjusted_away_goals: float
    home_win: float
    draw: float
    away_win: float
    predicted_home_goals: int
    predicted_away_goals: int
    home_fair_odds: float
    draw_fair_odds: float
    away_fair_odds: float
    home_expected_return: float | None
    draw_expected_return: float | None
    away_expected_return: float | None
    data_type: str

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return self.home_win, self.draw, self.away_win

    def to_record(self) -> dict[str, object]:
        home_pct, draw_pct, away_pct = percentages_summing_to_100(self.probabilities)
        return {
            "경기번호": self.match_id,
            "홈팀": self.home_team,
            "원정팀": self.away_team,
            "보정 홈득점": round(self.adjusted_home_goals, 3),
            "보정 원정득점": round(self.adjusted_away_goals, 3),
            "홈승 확률(%)": home_pct,
            "무승부 확률(%)": draw_pct,
            "원정승 확률(%)": away_pct,
            "예상 스코어": f"{self.predicted_home_goals}:{self.predicted_away_goals}",
            "홈승 적정배당": round(self.home_fair_odds, 3),
            "무승부 적정배당": round(self.draw_fair_odds, 3),
            "원정승 적정배당": round(self.away_fair_odds, 3),
            "홈승 기대수익률(%)": _round_optional(self.home_expected_return),
            "무승부 기대수익률(%)": _round_optional(self.draw_expected_return),
            "원정승 기대수익률(%)": _round_optional(self.away_expected_return),
            "데이터 구분": self.data_type,
        }


@dataclass(frozen=True)
class CombinationSearchResult:
    combinations: tuple[tuple[tuple[str, ...], float], ...]
    total_space: int
    expanded_candidates: int
    max_frontier_size: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "순위": rank,
                    "조합": "-".join(outcomes),
                    "조합 확률(%)": round(probability * 100.0, 8),
                }
                for rank, (outcomes, probability) in enumerate(
                    self.combinations, start=1
                )
            ]
        )


_COLUMN_ALIASES = {
    "match_id": "match_id",
    "경기번호": "match_id",
    "경기_번호": "match_id",
    "home_team": "home_team",
    "홈팀": "home_team",
    "away_team": "away_team",
    "원정팀": "away_team",
    "home_xg": "home_xg",
    "홈_기대득점": "home_xg",
    "홈기대득점": "home_xg",
    "away_xg": "away_xg",
    "원정_기대득점": "away_xg",
    "원정기대득점": "away_xg",
    "home_elo": "home_elo",
    "홈_elo": "home_elo",
    "홈elo": "home_elo",
    "away_elo": "away_elo",
    "원정_elo": "away_elo",
    "원정elo": "away_elo",
    "home_odds": "home_odds",
    "홈승_배당": "home_odds",
    "홈승배당": "home_odds",
    "draw_odds": "draw_odds",
    "무승부_배당": "draw_odds",
    "무승부배당": "draw_odds",
    "away_odds": "away_odds",
    "원정승_배당": "away_odds",
    "원정승배당": "away_odds",
    "data_type": "data_type",
    "데이터_구분": "data_type",
    "데이터구분": "data_type",
}

_REQUIRED_COLUMNS = ("home_team", "away_team", "home_xg", "away_xg")
_NUMERIC_COLUMNS = (
    "home_xg",
    "away_xg",
    "home_elo",
    "away_elo",
    "home_odds",
    "draw_odds",
    "away_odds",
)


def _normalise_header(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def read_match_csv(payload: bytes) -> pd.DataFrame:
    """업로드된 CSV 바이트를 UTF-8 또는 CP949로 읽는다."""

    if not payload:
        raise ValueError("CSV 파일이 비어 있습니다.")
    errors: list[str] = []
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(BytesIO(payload), encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949를 사용하세요.")


def normalise_match_dataframe(
    frame: pd.DataFrame, *, default_data_type: str = "사용자 업로드"
) -> pd.DataFrame:
    """한국어/영문 헤더를 내부 스키마로 변환하고 입력값을 검증한다."""

    if frame.empty:
        raise ValueError("분석할 경기 데이터가 없습니다.")

    rename_map: dict[object, str] = {}
    targets: set[str] = set()
    for column in frame.columns:
        key = _normalise_header(column)
        target = _COLUMN_ALIASES.get(key)
        if target is None:
            continue
        if target in targets:
            raise ValueError(f"같은 의미의 열이 중복되었습니다: {target}")
        targets.add(target)
        rename_map[column] = target

    result = frame.rename(columns=rename_map).copy()
    missing = [column for column in _REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

    row_count = len(result)
    defaults: dict[str, object] = {
        "match_id": [str(index) for index in range(1, row_count + 1)],
        "home_elo": 1500.0,
        "away_elo": 1500.0,
        "home_odds": np.nan,
        "draw_odds": np.nan,
        "away_odds": np.nan,
        "data_type": default_data_type,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    for column in ("home_team", "away_team"):
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or (result[column] == "").any():
            raise ValueError(f"{column}에 빈 팀 이름이 있습니다.")

    result["match_id"] = result["match_id"].astype("string").str.strip()
    result["data_type"] = (
        result["data_type"].astype("string").fillna(default_data_type).str.strip()
    )
    result.loc[result["data_type"] == "", "data_type"] = default_data_type

    for column in _NUMERIC_COLUMNS:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} 열에는 숫자만 입력할 수 있습니다.") from exc

    for column in ("home_xg", "away_xg"):
        values = result[column]
        if (~np.isfinite(values)).any() or (values <= 0).any() or (values > 10).any():
            raise ValueError(f"{column} 값은 0보다 크고 10 이하여야 합니다.")

    for column in ("home_elo", "away_elo"):
        values = result[column]
        if (~np.isfinite(values)).any() or (values < 800).any() or (values > 2800).any():
            raise ValueError(f"{column} 값은 800~2800 범위여야 합니다.")

    for column in ("home_odds", "draw_odds", "away_odds"):
        values = result[column]
        invalid = values.notna() & ((~np.isfinite(values)) | (values <= 1.0))
        if invalid.any():
            raise ValueError(f"{column} 값은 비워 두거나 1보다 커야 합니다.")

    ordered_columns = (
        "match_id",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "home_elo",
        "away_elo",
        "home_odds",
        "draw_odds",
        "away_odds",
        "data_type",
    )
    return result.loc[:, ordered_columns].reset_index(drop=True)


def elo_adjusted_goals(
    home_goals: float,
    away_goals: float,
    home_elo: float,
    away_elo: float,
    config: ModelConfig | None = None,
) -> tuple[float, float]:
    """Elo 기대승률을 작은 득점 배율로 변환한다."""

    config = config or ModelConfig()
    elo_difference = home_elo + config.home_advantage_elo - away_elo
    home_expectation = 1.0 / (1.0 + 10.0 ** (-elo_difference / 400.0))
    centred_strength = 2.0 * home_expectation - 1.0
    home_multiplier = math.exp(config.elo_goal_scale * centred_strength)
    away_multiplier = math.exp(-config.elo_goal_scale * centred_strength)
    return home_goals * home_multiplier, away_goals * away_multiplier


def poisson_probability(expected_goals: float, goals: int) -> float:
    if expected_goals < 0 or goals < 0:
        raise ValueError("기대득점과 득점은 음수일 수 없습니다.")
    return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)


def dixon_coles_factor(
    home_goals: int,
    away_goals: int,
    expected_home: float,
    expected_away: float,
    rho: float,
) -> float:
    """0-0, 0-1, 1-0, 1-1에 적용하는 Dixon-Coles 보정계수."""

    if (home_goals, away_goals) == (0, 0):
        return 1.0 - expected_home * expected_away * rho
    if (home_goals, away_goals) == (0, 1):
        return 1.0 + expected_home * rho
    if (home_goals, away_goals) == (1, 0):
        return 1.0 + expected_away * rho
    if (home_goals, away_goals) == (1, 1):
        return 1.0 - rho
    return 1.0


def score_probability_matrix(
    expected_home: float,
    expected_away: float,
    config: ModelConfig | None = None,
) -> np.ndarray:
    """절단된 스코어 행렬을 만들고 합계가 1이 되도록 정규화한다."""

    config = config or ModelConfig()
    home_probabilities = np.array(
        [
            poisson_probability(expected_home, goals)
            for goals in range(config.max_goals + 1)
        ],
        dtype=float,
    )
    away_probabilities = np.array(
        [
            poisson_probability(expected_away, goals)
            for goals in range(config.max_goals + 1)
        ],
        dtype=float,
    )
    matrix = np.outer(home_probabilities, away_probabilities)

    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[home_goals, away_goals] *= dixon_coles_factor(
            home_goals,
            away_goals,
            expected_home,
            expected_away,
            config.dixon_coles_rho,
        )

    if (matrix < 0).any():
        raise ValueError("Dixon-Coles 설정으로 음수 확률이 생성되었습니다.")
    total = float(matrix.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("유효한 스코어 확률을 계산할 수 없습니다.")
    return matrix / total


def analyze_match(
    row: Mapping[str, object], config: ModelConfig | None = None
) -> MatchAnalysis:
    config = config or ModelConfig()
    adjusted_home, adjusted_away = elo_adjusted_goals(
        float(row["home_xg"]),
        float(row["away_xg"]),
        float(row["home_elo"]),
        float(row["away_elo"]),
        config,
    )
    matrix = score_probability_matrix(adjusted_home, adjusted_away, config)
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    probability_total = home_win + draw + away_win
    home_win, draw, away_win = (
        home_win / probability_total,
        draw / probability_total,
        away_win / probability_total,
    )
    predicted_home, predicted_away = np.unravel_index(np.argmax(matrix), matrix.shape)

    home_odds = _optional_float(row.get("home_odds"))
    draw_odds = _optional_float(row.get("draw_odds"))
    away_odds = _optional_float(row.get("away_odds"))

    return MatchAnalysis(
        match_id=str(row["match_id"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        adjusted_home_goals=adjusted_home,
        adjusted_away_goals=adjusted_away,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        predicted_home_goals=int(predicted_home),
        predicted_away_goals=int(predicted_away),
        home_fair_odds=1.0 / home_win,
        draw_fair_odds=1.0 / draw,
        away_fair_odds=1.0 / away_win,
        home_expected_return=_expected_return(home_win, home_odds),
        draw_expected_return=_expected_return(draw, draw_odds),
        away_expected_return=_expected_return(away_win, away_odds),
        data_type=str(row["data_type"]),
    )


def analyze_dataframe(
    frame: pd.DataFrame,
    config: ModelConfig | None = None,
    *,
    default_data_type: str = "사용자 업로드",
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[MatchAnalysis, ...]]:
    """입력 프레임, 표시용 결과, 원시 분석 객체를 함께 반환한다."""

    config = config or ModelConfig()
    normalised = normalise_match_dataframe(
        frame, default_data_type=default_data_type
    )
    analyses = tuple(
        analyze_match(row, config) for row in normalised.to_dict(orient="records")
    )
    output = pd.DataFrame([analysis.to_record() for analysis in analyses])
    return normalised, output, analyses


def percentages_summing_to_100(
    probabilities: Sequence[float], decimals: int = 2
) -> tuple[float, ...]:
    """표시 반올림 후에도 합이 정확히 100이 되도록 최대 나머지법을 쓴다."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) == 0 or (values < 0).any():
        raise ValueError("확률은 비어 있지 않은 1차원 음이 아닌 값이어야 합니다.")
    total = float(values.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("확률 합계는 양의 유한값이어야 합니다.")
    values = values / total
    scale = 10**decimals
    raw_units = values * 100.0 * scale
    units = np.floor(raw_units).astype(int)
    missing_units = int(100 * scale - units.sum())
    order = np.argsort(-(raw_units - units), kind="stable")
    for index in order[:missing_units]:
        units[index] += 1
    return tuple(float(value / scale) for value in units)


def validate_toto_14(analyses: Sequence[MatchAnalysis]) -> None:
    if len(analyses) != 14:
        raise ValueError(f"승무패 분석에는 정확히 14경기가 필요합니다. 현재 {len(analyses)}경기입니다.")


def calculate_top_combinations(
    analyses: Sequence[MatchAnalysis], top_n: int = 20
) -> CombinationSearchResult:
    """매 단계 상위 N개만 유지해 전체 3^경기수 조합 생성을 피한다.

    이후 경기의 확률 곱 집합은 모든 부분 조합에 동일하므로, 같은 깊이에서
    상위 N개 밖의 부분 조합은 최종 상위 N개로 다시 진입할 수 없다.
    """

    if not analyses:
        raise ValueError("조합을 계산할 경기가 없습니다.")
    if top_n <= 0:
        raise ValueError("top_n은 1 이상이어야 합니다.")

    frontier: list[tuple[tuple[str, ...], float]] = [(tuple(), 1.0)]
    expanded_candidates = 0
    max_frontier_size = 1

    for analysis in analyses:
        probabilities = analysis.probabilities
        expanded = [
            (outcomes + (label,), probability * outcome_probability)
            for outcomes, probability in frontier
            for label, outcome_probability in zip(OUTCOMES, probabilities, strict=True)
        ]
        expanded_candidates += len(expanded)
        frontier = heapq.nlargest(top_n, expanded, key=lambda item: item[1])
        max_frontier_size = max(max_frontier_size, len(frontier))

    return CombinationSearchResult(
        combinations=tuple(frontier),
        total_space=3 ** len(analyses),
        expanded_candidates=expanded_candidates,
        max_frontier_size=max_frontier_size,
    )


def toto_insight_frames(
    analyses: Sequence[MatchAnalysis], limit: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """무승부 후보와 비우세 결과 합계가 큰 경기를 정리한다."""

    draw_rows: list[dict[str, object]] = []
    upset_rows: list[dict[str, object]] = []
    for analysis in analyses:
        probabilities = analysis.probabilities
        favourite_index = int(np.argmax(probabilities))
        alternative_indices = [index for index in range(3) if index != favourite_index]
        main_alternative = max(alternative_indices, key=lambda index: probabilities[index])
        draw_rows.append(
            {
                "경기번호": analysis.match_id,
                "경기": f"{analysis.home_team} vs {analysis.away_team}",
                "무승부 확률(%)": round(analysis.draw * 100.0, 2),
            }
        )
        upset_rows.append(
            {
                "경기번호": analysis.match_id,
                "경기": f"{analysis.home_team} vs {analysis.away_team}",
                "최우선 선택": OUTCOMES[favourite_index],
                "주요 대안": OUTCOMES[main_alternative],
                "비우세 결과 합계(%)": round((1.0 - probabilities[favourite_index]) * 100.0, 2),
            }
        )

    draw_frame = pd.DataFrame(draw_rows).sort_values(
        "무승부 확률(%)", ascending=False, ignore_index=True
    )
    upset_frame = pd.DataFrame(upset_rows).sort_values(
        "비우세 결과 합계(%)", ascending=False, ignore_index=True
    )
    return draw_frame.head(limit), upset_frame.head(limit)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _expected_return(probability: float, odds: float | None) -> float | None:
    if odds is None:
        return None
    return (probability * odds - 1.0) * 100.0


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
