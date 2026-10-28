from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from backtest_2026 import empty_round_status, evaluate, load_history, predict, prepare_round, run

ROOT = Path(__file__).resolve().parent


def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets, history = [], []
    for n in range(1, 15):
        home, away = f"가상 홈 {n}", f"가상 원정 {n}"
        targets.append({"year": 2026, "round": 1, "number": n, "competition": "TEST",
                        "kickoff": "2026-01-04 20:00", "deadline": "2026-01-04 19:00",
                        "home_team": home, "away_team": away, "home_id": n * 2, "away_id": n * 2 + 1,
                        "result_status": "정상", "home_score": 1, "away_score": 0, "outcome": "승",
                        "source_url": "demo://fictional", "verification": "확인"})
        history.append({"competition": "TEST", "kickoff": "2025-12-28 20:00", "finished_at": "2025-12-28 22:00",
                        "home_team": home, "away_team": away, "home_id": n * 2, "away_id": n * 2 + 1,
                        "home_score": 1, "away_score": 0, "source_url": "demo://fictional", "acquired_at": "2025-12-29 12:00"})
    return pd.DataFrame(targets), pd.DataFrame(history)


def test_prepare_14_sorted_separates_truth_and_checks_score() -> None:
    rounds, _ = data()
    prepared = prepare_round(rounds.sample(frac=1, random_state=1), 1)
    assert prepared.fixtures["number"].tolist() == list(range(1, 15))
    assert not {"outcome", "home_score", "away_score"} & set(prepared.fixtures.columns)
    rounds.loc[0, "outcome"] = "패"
    with pytest.raises(ValueError, match="결과 불일치"):
        prepare_round(rounds, 1)


def test_future_truth_cannot_change_prediction_and_late_results_excluded() -> None:
    rounds, history = data()
    a = predict(prepare_round(rounds, 1), load_history(history))
    rounds["home_score"] = 0
    rounds["away_score"] = 2
    rounds["outcome"] = "패"
    future = history.iloc[[0]].copy()
    future["kickoff"] = "2026-01-04 18:30"
    future["finished_at"] = "2026-01-04 20:30"
    future["acquired_at"] = "2026-01-05 12:00"
    future["home_score"] = 10
    with_future = pd.concat([history, future], ignore_index=True)
    b = predict(prepare_round(rounds, 1), load_history(with_future))
    assert a["predictions"] == b["predictions"]
    assert a["top20"] == b["top20"]


def test_acquisition_after_deadline_is_never_used_as_prior() -> None:
    rounds, history = data()
    history.loc[0, "acquired_at"] = "2026-01-05 12:00"
    with pytest.raises(ValueError, match="전력 기록 부족"):
        predict(prepare_round(rounds, 1), load_history(history))


def test_missing_acquisition_is_not_strictly_verified() -> None:
    _, history = data()
    history.loc[0, "acquired_at"] = ""
    with pytest.raises(ValueError, match="필수값 누락"):
        load_history(history)


def test_unknown_or_mismatched_team_and_special_round_excluded() -> None:
    rounds, history = data()
    rounds.loc[0, "home_team"] = "비슷한 다른 팀"
    with pytest.raises(ValueError, match="팀 ID와 팀명 불일치"):
        predict(prepare_round(rounds, 1), load_history(history))
    rounds, _ = data()
    rounds.loc[0, "result_status"] = "적중특례"
    with pytest.raises(ValueError, match="적중특례"):
        prepare_round(rounds, 1)


def test_bad_numbers_and_incomplete_rounds_never_pass() -> None:
    rounds, _ = data()
    rounds.loc[0, "number"] = 2
    with pytest.raises(ValueError, match="경기번호"):
        prepare_round(rounds, 1)
    with pytest.raises(ValueError, match="14경기 미확보"):
        prepare_round(rounds.iloc[:13], 1)


def test_cutoff_fallback_and_in_progress_history_exclusion() -> None:
    rounds, history = data()
    rounds["deadline"] = ""
    prepared = prepare_round(rounds, 1)
    assert prepared.cutoff.strftime("%Y-%m-%d %H:%M") == "2026-01-04 19:59"
    history.loc[0, "kickoff"] = "2026-01-04 19:30"
    history.loc[0, "finished_at"] = "2026-01-04 21:30"
    history.loc[0, "acquired_at"] = "2026-01-04 22:00"
    with pytest.raises(ValueError, match="전력 기록 부족"):
        predict(prepared, load_history(history))


def test_probability_and_top20_metrics() -> None:
    rounds, history = data()
    prepared = prepare_round(rounds, 1)
    prediction = predict(prepared, load_history(history))
    assert len(prediction["top20"]) == 20
    assert all(sum(p["probabilities"]) == pytest.approx(1.0) for p in prediction["predictions"])
    summary, matches, combinations = evaluate(prediction, prepared.outcomes, 1)
    first = prediction["predictions"][0]["probabilities"]
    assert matches[0]["Brier Score"] == pytest.approx((first[0] - 1)**2 + first[1]**2 + first[2]**2)
    assert matches[0]["Log Loss"] == pytest.approx(-math.log(first[0]))
    assert summary["단식 적중"] == f"{sum(m['적중'] for m in matches)}/14"
    assert summary["TOP20 최고 적중"] == f"{max(c['일치 경기수'] for c in combinations)}/14"
    corrupt = deepcopy(prediction)
    corrupt["predictions"][0]["home_team"] = "잘못된 홈팀"
    with pytest.raises(ValueError, match="홈원정"):
        evaluate(corrupt, prepared.outcomes, 1)


def test_batch_records_63_rounds_and_reuses_checkpoint(tmp_path: Path) -> None:
    rounds, history = data()
    output = run(rounds, history, tmp_path)
    assert len(output["rounds"]) == 63
    assert output["rounds"].loc[0, "모델 분석"] == "완료"
    assert output["rounds"].loc[1, "모델 분석"] == "미검증"
    assert len(output["matches"]) == 14
    snapshots = list((tmp_path / "checkpoints").glob("*.json"))
    assert len(snapshots) == 1
    assert '"outcome"' not in snapshots[0].read_text(encoding="utf-8")
    repeated = run(rounds, history, tmp_path)
    pd.testing.assert_frame_equal(output["matches"], repeated["matches"])
    rounds["home_score"] = 0
    rounds["away_score"] = 1
    rounds["outcome"] = "패"
    changed = run(rounds, history, tmp_path)
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 1
    assert changed["matches"]["AI 선택"].tolist() == output["matches"]["AI 선택"].tolist()
    assert len(empty_round_status()) == 63


def test_missing_inputs_do_not_invent_real_statistics(tmp_path: Path) -> None:
    tables = run(None, None, tmp_path)
    assert len(tables["errors"]) == 63
    assert tables["matches"].empty
    assert tables["top20"].empty
    assert "미검증" in (tmp_path / "backtest_2026_summary.md").read_text(encoding="utf-8")


def test_streamlit_shows_backtest_tab_without_external_data() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not list(app.exception)
    assert any(tab.label == "2026 승무패 백테스트" for tab in app.tabs)
