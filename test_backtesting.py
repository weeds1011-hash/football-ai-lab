from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from backtesting import (
    MODE_RECONSTRUCTED,
    PRODUCT_PROTO,
    PRODUCT_TOTO,
    BacktestContext,
    OutcomeSet,
    PredictionSnapshot,
    aggregate_evaluations,
    create_prediction_snapshot,
    evaluate_snapshot,
    normalise_backtest_dataframe,
    prepare_round,
    read_backtest_file,
    read_history_file,
    reconstructed_model_input,
    round_status_frame,
    run_snapshot_batch,
    snapshot_model_input,
)
from football_data import HistoricalMatch


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "data" / "demo_backtest_rounds.csv"


def _raw_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "상품": "프로토 승부식",
        "회차": "완료-1",
        "경기번호": 1,
        "종목": "축구",
        "대회": "테스트 리그",
        "대회코드": "PL",
        "경기시각(KST)": "2026-03-10 20:00",
        "마감시각(KST)": "2026-03-10 19:50",
        "홈팀": "Alpha FC",
        "원정팀": "Beta FC",
        "시장유형": "승무패",
        "홈득점": 2,
        "원정득점": 1,
        "홈팀 ID": 1,
        "원정팀 ID": 2,
        "홈 기대득점": 1.5,
        "원정 기대득점": 1.0,
        "홈 Elo": 1530,
        "원정 Elo": 1490,
        "스냅샷시각(KST)": "2026-03-10 19:00",
        "조회기준시각(KST)": "2026-10-27 12:00",
        "출처 URL": "demo://test",
        "데이터 구분": "가상 테스트 데이터",
    }
    row.update(updates)
    return row


def _history_match(
    match_id: str,
    kickoff: datetime,
    home_goals: int,
    away_goals: int,
    *,
    home_name: str = "Alpha FC",
    away_name: str = "Beta FC",
    home_id: int = 1,
    away_id: int = 2,
) -> HistoricalMatch:
    return HistoricalMatch(
        match_id=match_id,
        kickoff=kickoff,
        home_team_id=home_id,
        home_team=home_name,
        away_team_id=away_id,
        away_team=away_name,
        home_goals=home_goals,
        away_goals=away_goals,
        finished_at=kickoff + timedelta(hours=2),
    )


def test_demo_rounds_are_explicitly_fictional_complete_and_ordered() -> None:
    frame = read_backtest_file(DEMO.read_bytes(), "demo_backtest_rounds.csv")
    assert set(frame["data_type"]) == {"가상 백테스트 데이터"}
    status = round_status_frame(frame, PRODUCT_TOTO)
    assert len(status) == 5
    assert set(status["상태"]) == {"완료"}
    prepared = prepare_round(frame, PRODUCT_TOTO, "가상-01")
    assert prepared.context.target_frame["match_no"].tolist() == list(range(1, 15))
    assert "home_goals" not in prepared.context.target_frame
    assert "actual_outcome" not in prepared.context.target_frame
    assert prepared.outcomes.frame["match_no"].tolist() == list(range(1, 15))


def test_cutoff_prefers_common_deadline_and_falls_back_before_first_kickoff() -> None:
    with_deadline = normalise_backtest_dataframe(pd.DataFrame([_raw_row()]))
    prepared = prepare_round(with_deadline, PRODUCT_PROTO, "완료-1")
    assert prepared.context.cutoff.strftime("%Y-%m-%d %H:%M") == "2026-03-10 19:50"
    assert prepared.context.cutoff_basis == "회차 공통 마감시각"

    without = normalise_backtest_dataframe(
        pd.DataFrame([_raw_row(**{"마감시각(KST)": ""})])
    )
    fallback = prepare_round(without, PRODUCT_PROTO, "완료-1")
    assert fallback.context.cutoff.strftime("%Y-%m-%d %H:%M") == "2026-03-10 19:59"
    assert "첫 경기 시작 1분 전" in fallback.context.cutoff_basis


def test_incomplete_round_and_non_fourteen_toto_are_rejected() -> None:
    missing_result = normalise_backtest_dataframe(
        pd.DataFrame([_raw_row(**{"홈득점": ""})])
    )
    with pytest.raises(ValueError, match="실제 결과가 없는"):
        prepare_round(missing_result, PRODUCT_PROTO, "완료-1")

    rows = [
        _raw_row(
            **{
                "상품": "축구토토 승무패",
                "회차": "13경기",
                "경기번호": number,
                "홈팀": f"홈 {number}",
                "원정팀": f"원정 {number}",
            }
        )
        for number in range(1, 14)
    ]
    with pytest.raises(ValueError, match="정확히 14경기"):
        prepare_round(
            normalise_backtest_dataframe(pd.DataFrame(rows)),
            PRODUCT_TOTO,
            "13경기",
        )


def test_actual_scores_cannot_change_prediction_snapshot() -> None:
    first = normalise_backtest_dataframe(pd.DataFrame([_raw_row()]))
    changed = normalise_backtest_dataframe(
        pd.DataFrame([_raw_row(**{"홈득점": 0, "원정득점": 5})])
    )
    prepared_first = prepare_round(first, PRODUCT_PROTO, "완료-1")
    prepared_changed = prepare_round(changed, PRODUCT_PROTO, "완료-1")
    pd.testing.assert_frame_equal(
        prepared_first.context.target_frame,
        prepared_changed.context.target_frame,
    )
    snapshot_first = create_prediction_snapshot(
        prepared_first.context, snapshot_model_input(prepared_first.context)
    )
    snapshot_changed = create_prediction_snapshot(
        prepared_changed.context, snapshot_model_input(prepared_changed.context)
    )
    assert snapshot_first.snapshot_id == snapshot_changed.snapshot_id
    pd.testing.assert_frame_equal(snapshot_first.predictions, snapshot_changed.predictions)
    assert "actual_outcome" not in prepared_first.outcomes.frame
    assert evaluate_snapshot(snapshot_first, prepared_first.outcomes).match_frame.loc[0, "실제"] == "승"
    assert evaluate_snapshot(snapshot_changed, prepared_changed.outcomes).match_frame.loc[0, "실제"] == "패"


def test_snapshot_recorded_after_cutoff_is_rejected() -> None:
    frame = normalise_backtest_dataframe(
        pd.DataFrame([_raw_row(**{"스냅샷시각(KST)": "2026-03-10 19:55"})])
    )
    context = prepare_round(frame, PRODUCT_PROTO, "완료-1").context
    with pytest.raises(ValueError, match="기준시각 이후"):
        snapshot_model_input(context)


def test_reconstruction_excludes_future_results_and_cutoff_changes_history() -> None:
    frame = normalise_backtest_dataframe(pd.DataFrame([_raw_row()]))
    context = prepare_round(frame, PRODUCT_PROTO, "완료-1").context
    cutoff = context.cutoff.tz_convert(timezone.utc).to_pydatetime()
    past = _history_match("past", cutoff - timedelta(days=3), 1, 0)
    future_a = _history_match("future", cutoff + timedelta(hours=1), 9, 0)
    future_b = _history_match("future", cutoff + timedelta(hours=1), 0, 9)

    baseline = reconstructed_model_input(context, {"PL": [past, future_a]})
    changed_future = reconstructed_model_input(context, {"PL": [past, future_b]})
    pd.testing.assert_frame_equal(baseline.frame, changed_future.frame)
    assert baseline.history_match_count == 1

    later_targets = context.target_frame.copy()
    later_targets["kickoff_kst"] = later_targets["kickoff_kst"].map(
        lambda value: value.to_pydatetime() + timedelta(days=1)
    )
    later_context = replace(
        context,
        cutoff=pd.Timestamp(cutoff + timedelta(hours=4)).tz_convert("Asia/Seoul"),
        target_frame=later_targets,
    )
    later = reconstructed_model_input(later_context, {"PL": [past, future_a]})
    assert later.history_match_count == 2
    assert not baseline.frame.equals(later.frame)


def test_team_matching_is_exact_and_ambiguous_names_are_rejected() -> None:
    without_ids = normalise_backtest_dataframe(
        pd.DataFrame([_raw_row(**{"홈팀 ID": "", "원정팀 ID": ""})])
    )
    context = prepare_round(without_ids, PRODUCT_PROTO, "완료-1").context
    cutoff = context.cutoff.tz_convert(timezone.utc).to_pydatetime()
    history = [_history_match("past", cutoff - timedelta(days=1), 1, 0)]
    assert reconstructed_model_input(context, {"PL": history}).mode == MODE_RECONSTRUCTED

    similar = context.target_frame.copy()
    similar.loc[0, "home_team"] = "Alpha"
    with pytest.raises(ValueError, match="완전 일치"):
        reconstructed_model_input(replace(context, target_frame=similar), {"PL": history})

    ambiguous_history = history + [
        _history_match(
            "duplicate-name",
            cutoff - timedelta(hours=12),
            0,
            0,
            home_name="Alpha FC",
            away_name="Other FC",
            home_id=99,
            away_id=3,
        )
    ]
    with pytest.raises(ValueError, match="2개"):
        reconstructed_model_input(context, {"PL": ambiguous_history})


def test_match_started_before_cutoff_but_finished_after_is_excluded() -> None:
    frame = normalise_backtest_dataframe(pd.DataFrame([_raw_row()]))
    context = prepare_round(frame, PRODUCT_PROTO, "완료-1").context
    cutoff = context.cutoff.tz_convert(timezone.utc).to_pydatetime()
    prior = _history_match("prior", cutoff - timedelta(days=3), 1, 0)
    in_progress = _history_match("in_progress", cutoff - timedelta(minutes=30), 9, 0)
    result = reconstructed_model_input(context, {"PL": [prior, in_progress]})
    assert result.history_match_count == 1


def test_history_csv_requires_result_timestamp_and_records_source() -> None:
    rows = [{"대회코드": "PL", "경기시작(KST)": "2026-03-07 12:00",
             "결과확정시각(KST)": "2026-03-07 14:00", "홈팀 ID": 1,
             "홈팀": "Alpha FC", "원정팀 ID": 2, "원정팀": "Beta FC",
             "홈득점": 1, "원정득점": 0, "출처": "가상 이력",
             "확보시각(KST)": "2026-10-27 12:00"}]
    data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
    history = read_history_file(data, "history.csv")
    context = prepare_round(normalise_backtest_dataframe(pd.DataFrame([_raw_row()])),
                            PRODUCT_PROTO, "완료-1").context
    result = reconstructed_model_input(context, history)
    assert "가상 이력" in result.history_sources
    rows[0]["결과확정시각(KST)"] = ""
    with pytest.raises(ValueError, match="필수 정보"):
        read_history_file(pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), "history.csv")


def test_streamlit_predictions_frozen_before_reveal_and_batch_works() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not list(app.exception)
    app.radio(key="bt_product").set_value(PRODUCT_TOTO).run(timeout=30)
    app.button(key="bt_predict").click().run(timeout=30)
    assert not list(app.exception)
    frozen_id = app.session_state["bt_frozen"]["snapshot"].snapshot_id
    assert "bt_evaluation" not in app.session_state
    app.button(key="bt_reveal").click().run(timeout=30)
    assert not list(app.exception)
    assert app.session_state["bt_evaluation"]["result"].snapshot_id == frozen_id
    app.button(key="bt_batch_run").click().run(timeout=30)
    assert not list(app.exception)
    assert len(app.session_state["bt_batch"]["evaluations"]) == 5


def test_brier_score_and_log_loss_use_multiclass_probabilities() -> None:
    predictions = pd.DataFrame(
        [
            {
                "round_id": "metric",
                "product": PRODUCT_PROTO,
                "match_no": 1,
                "match_date": "2026-03-10",
                "kickoff_kst": "2026-03-10 20:00",
                "league": "테스트",
                "home_team": "Alpha FC",
                "away_team": "Beta FC",
                "home_probability": 0.7,
                "draw_probability": 0.2,
                "away_probability": 0.1,
                "predicted_outcome": "승",
                "predicted_score": "1:0",
            }
        ]
    )
    snapshot = PredictionSnapshot(
        snapshot_id="metric",
        product=PRODUCT_PROTO,
        round_id="metric",
        cutoff=pd.Timestamp("2026-03-10 19:50", tz="Asia/Seoul"),
        cutoff_basis="테스트",
        model_version="test",
        mode="test",
        source_label="test",
        acquired_at_label="test",
        created_at=pd.Timestamp("2026-03-10", tz="UTC"),
        history_match_count=0,
        history_start="test",
        history_end="test",
        predictions=predictions,
        top_combinations=None,
    )
    outcomes = OutcomeSet(
        product=PRODUCT_PROTO,
        round_id="metric",
        frame=pd.DataFrame(
            [
                {
                    "match_no": 1,
                    "home_team": "Alpha FC",
                    "away_team": "Beta FC",
                    "home_goals": 2,
                    "away_goals": 0,
                    "actual_outcome": "승",
                }
            ]
        ),
    )
    result = evaluate_snapshot(snapshot, outcomes)
    assert result.brier_score == pytest.approx((0.7 - 1) ** 2 + 0.2**2 + 0.1**2)
    assert result.log_loss == pytest.approx(-math.log(0.7))
    assert result.accuracy == 1.0


def test_toto_top20_comparison_and_all_fourteen_match() -> None:
    frame = read_backtest_file(DEMO.read_bytes(), "demo.csv")
    prepared = prepare_round(frame, PRODUCT_TOTO, "가상-01")
    snapshot = create_prediction_snapshot(
        prepared.context, snapshot_model_input(prepared.context)
    )
    top1 = snapshot.top_combinations.combinations[0][0]
    perfect = prepared.outcomes.frame.copy()
    perfect["actual_outcome"] = list(top1)
    perfect["home_goals"] = [1 if label == "승" else 0 for label in top1]
    perfect["away_goals"] = [1 if label == "패" else 0 for label in top1]
    result = evaluate_snapshot(
        snapshot,
        OutcomeSet(PRODUCT_TOTO, "가상-01", perfect),
    )
    assert result.actual_in_top20 is True
    assert result.actual_rank == 1
    assert result.top1_matching_matches == 14
    assert result.all_14_correct is True


def test_five_round_batch_aggregation_keeps_round_trend() -> None:
    frame = read_backtest_file(DEMO.read_bytes(), "demo.csv")
    round_ids = [f"가상-{number:02d}" for number in range(1, 6)]
    evaluations = run_snapshot_batch(frame, PRODUCT_TOTO, round_ids)
    aggregate = aggregate_evaluations(evaluations)
    assert len(evaluations) == 5
    assert aggregate.total_matches == 70
    assert len(aggregate.trend_frame) == 5
    assert 0.0 <= aggregate.accuracy <= 1.0
