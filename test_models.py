from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from models import (
    ModelConfig,
    analyze_dataframe,
    analyze_match,
    calculate_top_combinations,
    elo_adjusted_goals,
    normalise_match_dataframe,
    percentages_summing_to_100,
    read_match_csv,
    score_probability_matrix,
    validate_toto_14,
)


ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "data" / "demo_14_matches.csv"


def _base_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": "1",
        "home_team": "가상 홈",
        "away_team": "가상 원정",
        "home_xg": 1.5,
        "away_xg": 1.1,
        "home_elo": 1500,
        "away_elo": 1500,
        "home_odds": 2.2,
        "draw_odds": 3.2,
        "away_odds": 3.4,
        "data_type": "가상 데이터",
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    ("home_xg", "away_xg", "home_elo", "away_elo"),
    [(1.5, 1.1, 1500, 1500), (0.7, 0.6, 1400, 1550), (2.8, 1.9, 1800, 1600)],
)
def test_match_probabilities_sum_to_one(
    home_xg: float, away_xg: float, home_elo: float, away_elo: float
) -> None:
    result = analyze_match(
        _base_row(
            home_xg=home_xg,
            away_xg=away_xg,
            home_elo=home_elo,
            away_elo=away_elo,
        )
    )
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 < value < 1.0 for value in result.probabilities)


def test_display_percentages_sum_exactly_100() -> None:
    values = percentages_summing_to_100((0.333333, 0.333333, 0.333334))
    assert sum(values) == pytest.approx(100.0)
    assert values == (33.33, 33.33, 33.34)


def test_dixon_coles_changes_only_low_score_cells_before_normalisation() -> None:
    independent = score_probability_matrix(
        1.3, 1.1, ModelConfig(dixon_coles_rho=0.0)
    )
    corrected = score_probability_matrix(
        1.3, 1.1, ModelConfig(dixon_coles_rho=-0.08)
    )
    assert corrected.sum() == pytest.approx(1.0)
    assert not np.allclose(independent[:2, :2], corrected[:2, :2])


def test_elo_advantage_increases_home_goal_expectation() -> None:
    balanced_home, balanced_away = elo_adjusted_goals(1.4, 1.2, 1500, 1500)
    strong_home, strong_away = elo_adjusted_goals(1.4, 1.2, 1800, 1400)
    assert strong_home > balanced_home
    assert strong_away < balanced_away


def test_fair_odds_and_expected_return_formula() -> None:
    result = analyze_match(_base_row())
    assert result.home_fair_odds == pytest.approx(1 / result.home_win)
    assert result.home_expected_return == pytest.approx(
        (result.home_win * 2.2 - 1) * 100
    )


def test_korean_csv_schema_and_cp949_upload() -> None:
    source = pd.DataFrame(
        {
            "홈팀": ["가상 A"],
            "원정팀": ["가상 B"],
            "홈 기대득점": [1.2],
            "원정 기대득점": [0.9],
        }
    )
    payload = source.to_csv(index=False).encode("cp949")
    decoded = read_match_csv(payload)
    normalised = normalise_match_dataframe(decoded)
    assert normalised.loc[0, "home_team"] == "가상 A"
    assert normalised.loc[0, "home_elo"] == 1500
    assert normalised.loc[0, "data_type"] == "사용자 업로드"


def test_invalid_odds_are_rejected() -> None:
    source = pd.DataFrame(
        {
            "홈팀": ["가상 A"],
            "원정팀": ["가상 B"],
            "홈 기대득점": [1.2],
            "원정 기대득점": [0.9],
            "홈승 배당": [1.0],
        }
    )
    with pytest.raises(ValueError, match="1보다 커야"):
        normalise_match_dataframe(source)


def test_demo_has_14_explicitly_fictional_matches() -> None:
    source = read_match_csv(DEMO_CSV.read_bytes())
    normalised, output, analyses = analyze_dataframe(source)
    validate_toto_14(analyses)
    assert len(normalised) == len(output) == 14
    assert set(normalised["data_type"]) == {"가상 데이터"}
    probability_columns = ["홈승 확률(%)", "무승부 확률(%)", "원정승 확률(%)"]
    assert np.allclose(output[probability_columns].sum(axis=1), 100.0)


def test_validate_toto_requires_exactly_14_matches() -> None:
    analysis = analyze_match(_base_row())
    with pytest.raises(ValueError, match="정확히 14경기"):
        validate_toto_14([analysis] * 13)


def test_top_combinations_match_brute_force_for_small_case() -> None:
    analyses = tuple(
        analyze_match(_base_row(match_id=str(index), home_xg=1.0 + index * 0.1))
        for index in range(1, 5)
    )
    result = calculate_top_combinations(analyses, top_n=20)

    brute_force = []
    labels = ("승", "무", "패")
    for indices in product(range(3), repeat=len(analyses)):
        probability = np.prod(
            [analyses[index].probabilities[outcome] for index, outcome in enumerate(indices)]
        )
        brute_force.append((tuple(labels[index] for index in indices), probability))
    brute_force.sort(key=lambda item: item[1], reverse=True)

    assert [item[0] for item in result.combinations] == [
        item[0] for item in brute_force[:20]
    ]
    assert [item[1] for item in result.combinations] == pytest.approx(
        [item[1] for item in brute_force[:20]]
    )


def test_14_match_search_keeps_bounded_frontier() -> None:
    source = read_match_csv(DEMO_CSV.read_bytes())
    _, _, analyses = analyze_dataframe(source)
    result = calculate_top_combinations(analyses, top_n=20)
    assert result.total_space == 3**14
    assert len(result.combinations) == 20
    assert result.max_frontier_size <= 20
    assert result.expanded_candidates <= 14 * 20 * 3
    assert all(
        result.combinations[index][1] >= result.combinations[index + 1][1]
        for index in range(len(result.combinations) - 1)
    )


def test_streamlit_demo_buttons_complete_without_exception() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not list(app.exception)

    app.button(key="proto_run").click().run(timeout=30)
    assert not list(app.exception)
    assert len(app.session_state["proto_result"]) == 14

    app.button(key="toto_run").click().run(timeout=30)
    assert not list(app.exception)
    combination_search = app.session_state["toto_bundle"][1]
    assert len(combination_search.combinations) == 20
    assert combination_search.max_frontier_size == 20
