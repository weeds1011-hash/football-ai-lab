"""2026 승무패 사용자 제공 자료 백테스트. 자동 수집 기능은 없다.

회차 파일은 결과·경기표를 포함하고 이력 파일은 별도다. 예측 함수에는 회차의
정답 열을 넘기지 않으며 예측을 체크포인트에 저장한 뒤에만 정답을 결합한다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from football_data import HistoricalMatch, calculate_elo_ratings, calculate_team_form, estimate_expected_goals
from models import OUTCOMES, ModelConfig, analyze_dataframe, calculate_top_combinations

MODEL_VERSION = "elo-poisson-dixon-coles-v1"
TEMPORAL_POLICY_VERSION = "finish-and-availability-v2"
ROUND_COLUMNS = ["year", "round", "number", "competition", "kickoff", "deadline",
                 "home_team", "away_team", "home_id", "away_id", "result_status",
                 "home_score", "away_score", "outcome", "source_url", "verification"]
HISTORY_COLUMNS = ["competition", "kickoff", "finished_at", "home_team", "away_team",
                   "home_id", "away_id", "home_score", "away_score", "source_url", "acquired_at"]
ROUND_REPORT_COLUMNS = ["회차", "14경기 확보", "모델 분석", "단식 적중", "TOP20 최고 적중", "TOP20 정답 포함", "사유"]
MATCH_REPORT_COLUMNS = ["회차", "번호", "리그", "날짜", "홈팀", "원정팀", "홈승%", "무%", "원정승%", "AI 선택", "두번째", "예상 스코어", "실제 스코어", "실제 결과", "적중", "홈승 기준선 적중", "과거 빈도 기준선", "빈도 기준선 적중", "Brier Score", "Log Loss", "전력 자료", "가정"]
TOP_REPORT_COLUMNS = ["회차", "순위", "조합", "확률", "일치 경기수", "일치 등급"]


def _required(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} 필수 열 누락: {', '.join(missing)}")


def _time(value: Any) -> pd.Timestamp:
    if pd.isna(value) or not str(value).strip():
        raise ValueError("시간 정보가 없습니다.")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("시간 정보가 없습니다.")
    return stamp.tz_localize("Asia/Seoul") if stamp.tzinfo is None else stamp.tz_convert("Asia/Seoul")


def _score(row: pd.Series) -> tuple[int, int]:
    try:
        a, b = float(row["home_score"]), float(row["away_score"])
        if not np.isfinite([a, b]).all() or not a.is_integer() or not b.is_integer() or min(a, b) < 0:
            raise ValueError
        return int(a), int(b)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("정규시간 스코어가 없거나 잘못됐습니다.") from exc


def _outcome(a: int, b: int) -> str:
    return "승" if a > b else "패" if a < b else "무"


def load_history(frame: pd.DataFrame) -> dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]]:
    _required(frame, HISTORY_COLUMNS, "이력")
    groups: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]] = {}
    seen: set[tuple[object, ...]] = set()
    for index, row in frame.iterrows():
        if any(pd.isna(row[col]) or not str(row[col]).strip() for col in HISTORY_COLUMNS):
            raise ValueError(f"이력 {index + 1}행 필수값 누락")
        kick, finish, acquired = _time(row["kickoff"]), _time(row["finished_at"]), _time(row["acquired_at"])
        if finish <= kick or acquired < finish:
            raise ValueError(f"이력 {index + 1}행 시간 순서 오류")
        a, b = _score(row)
        try:
            hid, aid = int(row["home_id"]), int(row["away_id"])
            if hid <= 0 or aid <= 0 or hid == aid or float(row["home_id"]) != hid or float(row["away_id"]) != aid:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"이력 {index + 1}행 팀 ID 오류") from exc
        code = str(row["competition"]).strip().upper()
        key = (code, kick.isoformat(), hid, aid)
        if key in seen:
            raise ValueError(f"이력 {index + 1}행 경기 중복")
        seen.add(key)
        match = HistoricalMatch(str(index), kick.to_pydatetime(), hid, str(row["home_team"]).strip(),
                                aid, str(row["away_team"]).strip(), a, b)
        # 이력 결과가 확정되고 해당 자료가 확보된 이후에만 이용 가능하다.
        groups.setdefault(code, []).append((match, max(finish, acquired)))
    return groups


@dataclass(frozen=True)
class Prepared:
    fixtures: pd.DataFrame
    outcomes: pd.DataFrame
    cutoff: pd.Timestamp
    basis: str


def prepare_round(frame: pd.DataFrame, number: int) -> Prepared:
    _required(frame, ROUND_COLUMNS, "회차")
    years = pd.to_numeric(frame["year"], errors="coerce")
    rounds = pd.to_numeric(frame["round"], errors="coerce")
    group = frame.loc[(years == 2026) & (rounds == number)].copy()
    if len(group) != 14:
        raise ValueError(f"14경기 미확보: {len(group)}경기")
    nums = pd.to_numeric(group["number"], errors="coerce")
    if nums.isna().any() or sorted(nums.tolist()) != list(range(1, 15)):
        raise ValueError("공식 경기번호 1~14 불일치")
    group["number"] = nums.astype(int)
    group = group.sort_values("number").reset_index(drop=True)
    for col in ("competition", "home_team", "away_team", "home_id", "away_id", "source_url", "verification", "result_status"):
        if group[col].isna().any() or group[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} 누락")
    if not group["verification"].eq("확인").all():
        raise ValueError("출처·홈원정·경기 정합성 미확인")
    if not group["result_status"].eq("정상").all():
        raise ValueError("취소·연기·적중특례 또는 결과 상태 미확인")
    for _, row in group.iterrows():
        a, b = _score(row)
        if row["outcome"] != _outcome(a, b):
            raise ValueError(f"{row['number']}번 정규시간 점수와 결과 불일치")
    kicks = group["kickoff"].map(_time)
    first = min(kicks)
    deadlines = [_time(value) for value in group["deadline"] if not pd.isna(value) and str(value).strip()]
    if len(deadlines) == 14 and len(set(deadlines)) == 1:
        cutoff, basis = deadlines[0], "공통 마감시각"
    else:
        cutoff = min([pd.Timestamp(first.to_pydatetime() - timedelta(minutes=1)), *deadlines])
        basis = "마감 불완전·불일치: 첫 경기 1분 전과 기재 마감 중 이른 시각"
    if cutoff == first:
        cutoff = cutoff - pd.Timedelta(seconds=1)
        basis += " · 첫 경기와 동시각이므로 1초 전 적용"
    if cutoff > first:
        raise ValueError("회차 기준시각은 첫 경기 시작 전이어야 합니다.")
    # 예측 입력에는 실제 스코어, 결과 및 정답 상태를 포함하지 않는다.
    fixtures = group[["number", "competition", "kickoff", "home_team", "away_team", "home_id", "away_id"]].copy()
    outcomes = group[["number", "home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
    return Prepared(fixtures, outcomes, cutoff, basis)


def predict(prepared: Prepared, history: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]]) -> dict[str, Any]:
    model_rows, metadata = [], []
    for _, row in prepared.fixtures.iterrows():
        code = str(row["competition"]).strip().upper()
        prior = [m for m, finish in history.get(code, []) if finish < prepared.cutoff and m.kickoff < prepared.cutoff.to_pydatetime()]
        try:
            hid, aid = int(row["home_id"]), int(row["away_id"])
            if hid <= 0 or aid <= 0 or hid == aid or float(row["home_id"]) != hid or float(row["away_id"]) != aid:
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{row['number']}번 팀 ID 오류") from exc
        home_events = [m for m in prior if hid in (m.home_team_id, m.away_team_id)]
        away_events = [m for m in prior if aid in (m.home_team_id, m.away_team_id)]
        if not home_events or not away_events:
            raise ValueError(f"{row['number']}번 {code} 기준시각 이전 팀 전력 기록 부족")
        def names_for(team_id: int) -> set[str]:
            return {_team_name(name) for m in prior for candidate, name in
                    ((m.home_team_id, m.home_team), (m.away_team_id, m.away_team)) if candidate == team_id}
        def ids_for(name: object) -> set[int]:
            return {candidate for m in prior for candidate, candidate_name in
                    ((m.home_team_id, m.home_team), (m.away_team_id, m.away_team))
                    if _team_name(candidate_name) == _team_name(name)}
        if _team_name(row["home_team"]) not in names_for(hid) or _team_name(row["away_team"]) not in names_for(aid):
            raise ValueError(f"{row['number']}번 팀 ID와 팀명 불일치 · 확인된 이름 필요")
        if ids_for(row["home_team"]) != {hid} or ids_for(row["away_team"]) != {aid}:
            raise ValueError(f"{row['number']}번 팀명에 복수 ID가 있어 사용자 확인 필요")
        ratings = calculate_elo_ratings(prior)
        hg, ag = estimate_expected_goals(prior, home_id=hid, away_id=aid)
        hf = calculate_team_form(prior, team_id=hid, team_name=str(row["home_team"]), elo=ratings[hid], prediction_venue="home")
        af = calculate_team_form(prior, team_id=aid, team_name=str(row["away_team"]), elo=ratings[aid], prediction_venue="away")
        model_rows.append({"경기번호": str(int(row["number"])), "홈팀": row["home_team"], "원정팀": row["away_team"],
                           "홈 기대득점": hg, "원정 기대득점": ag, "홈 Elo": ratings[hid], "원정 Elo": ratings[aid], "데이터 구분": "사후 시점 재현"})
        counts = {label: sum(_outcome(m.home_goals, m.away_goals) == label for m in prior) for label in OUTCOMES}
        baseline = max(OUTCOMES, key=lambda label: counts[label])
        metadata.append({"competition": code, "date": _time(row["kickoff"]).date().isoformat(),
                         "history_count": len(prior), "home_matches": hf.matches_available, "away_matches": af.matches_available,
                         "uncertainty": "높음" if "높음" in (hf.uncertainty, af.uncertainty) else "보통 이하", "frequency_pick": baseline})
    _, _, analyses = analyze_dataframe(pd.DataFrame(model_rows), ModelConfig())
    combos = calculate_top_combinations(analyses, top_n=20)
    rows = []
    for analysis, meta in zip(analyses, metadata, strict=True):
        order = sorted(range(3), key=lambda i: (-analysis.probabilities[i], i))
        rows.append({"number": int(analysis.match_id), "home_team": analysis.home_team, "away_team": analysis.away_team,
                     "probabilities": list(analysis.probabilities), "pick": OUTCOMES[order[0]], "second": OUTCOMES[order[1]],
                     "score": f"{analysis.predicted_home_goals}:{analysis.predicted_away_goals}", **meta})
    config = ModelConfig()
    return {"model_version": MODEL_VERSION, "model_config": {"max_goals": config.max_goals,
            "dixon_coles_rho": config.dixon_coles_rho, "home_advantage_elo": config.home_advantage_elo,
            "elo_goal_scale": config.elo_goal_scale}, "created_at": datetime.now(timezone.utc).isoformat(),
            "cutoff": prepared.cutoff.isoformat(), "cutoff_basis": prepared.basis,
            "temporal_policy": TEMPORAL_POLICY_VERSION,
            "availability_verification": "사용자 제공 종료·확보시각 기준; 출처의 실제 당시 공개 여부는 별도 검증 필요",
            "predictions": rows, "top20": [{"outcomes": list(labels), "probability": prob} for labels, prob in combos.combinations],
            "assumption": "Elo 초기값 1500; 리그 및 홈/원정 평균 수축; 현 모델 고정 설정"}


def _team_name(value: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", str(value)).casefold())


def evaluate(snapshot: dict[str, Any], outcomes: pd.DataFrame, round_number: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    answers = outcomes.sort_values("number").to_dict("records")
    preds = snapshot["predictions"]
    if len(preds) != 14 or len(answers) != 14:
        raise ValueError("14경기 예측 또는 정답 누락")
    matches = []
    for prediction, answer in zip(preds, answers, strict=True):
        if (prediction["number"], prediction["home_team"], prediction["away_team"]) != (answer["number"], answer["home_team"], answer["away_team"]):
            raise ValueError("예측과 결과의 번호·홈원정 불일치")
        truth = answer["outcome"]
        probs = prediction["probabilities"]
        if len(probs) != 3 or any(not 0 <= p <= 1 for p in probs) or not np.isclose(sum(probs), 1):
            raise ValueError("예측 확률 오류")
        actual_index = OUTCOMES.index(truth)
        brier = sum((p - (i == actual_index)) ** 2 for i, p in enumerate(probs))
        logloss = -float(np.log(max(probs[actual_index], 1e-15)))
        matches.append({"회차": round_number, "번호": prediction["number"], "리그": prediction["competition"], "날짜": prediction["date"],
                        "홈팀": prediction["home_team"], "원정팀": prediction["away_team"], "홈승%": probs[0]*100, "무%": probs[1]*100, "원정승%": probs[2]*100,
                        "AI 선택": prediction["pick"], "두번째": prediction["second"], "예상 스코어": prediction["score"],
                        "실제 스코어": f"{int(answer['home_score'])}:{int(answer['away_score'])}", "실제 결과": truth,
                        "적중": prediction["pick"] == truth, "홈승 기준선 적중": truth == "승", "과거 빈도 기준선": prediction["frequency_pick"],
                        "빈도 기준선 적중": prediction["frequency_pick"] == truth, "Brier Score": brier, "Log Loss": logloss,
                        "전력 자료": f"홈 {prediction['home_matches']} / 원정 {prediction['away_matches']} 경기 · 불확실성 {prediction['uncertainty']}", "가정": snapshot["assumption"]})
    actual = [row["outcome"] for row in answers]
    tops = []
    for rank, combo in enumerate(snapshot["top20"], 1):
        score = sum(a == b for a, b in zip(combo["outcomes"], actual, strict=True))
        tops.append({"회차": round_number, "순위": rank, "조합": "-".join(combo["outcomes"]), "확률": combo["probability"],
                     "일치 경기수": score, "일치 등급": str(score) if score >= 11 else "10 이하"})
    correct = sum(row["적중"] for row in matches)
    return ({"회차": round_number, "14경기 확보": "확보", "모델 분석": "완료", "단식 적중": f"{correct}/14",
             "TOP20 최고 적중": f"{max(row['일치 경기수'] for row in tops)}/14", "TOP20 정답 포함": "예" if any(row["일치 경기수"] == 14 for row in tops) else "아니오", "사유": ""}, matches, tops)


def empty_round_status() -> pd.DataFrame:
    rows = [{"회차": n, "14경기 확보": "미확보", "모델 분석": "미검증", "단식 적중": "", "TOP20 최고 적중": "",
             "TOP20 정답 포함": "", "사유": "이용 권한 및 검증 상태가 확인된 회차·경기 전 이력 파일 미확보"}
            for n in range(1, 64)]
    rows[62]["14경기 확보"] = "공식 조회·미입력"
    rows[62]["사유"] = "공식 14경기 화면 확인; 분석용 파일의 이용 권한 및 경기 전 이력 미확보"
    return pd.DataFrame(rows, columns=ROUND_REPORT_COLUMNS)


def write_summary(result: dict[str, pd.DataFrame], output: Path) -> None:
    rounds, matches, tops = result["rounds"], result["matches"], result["top20"]
    completed = rounds.loc[rounds["모델 분석"] == "완료"]
    lines = ["# 2026 축구토토 승무패 백테스트 현황", "",
             "2026-10-27 기준 조사 범위: 1~63회차. 베트맨 공개 화면에서 63회차 결과 발표 및 14경기 결과, 64회차 발매 표시를 확인했다. 결과의 분석·재배포 권한은 확인되지 않았다.",
             f"분석 완료: {len(completed)}/63회차, {len(matches)}경기.",
             "과거 경기 전 시점의 사후 재구성이며 원래 저장된 당시 예측이 아니다.", ""]
    if matches.empty:
        lines += ["## 분석 결과", "", "실제 회차·전력 이력이 없어 적중률, 기준선, Brier Score, Log Loss는 미검증이다.", ""]
    else:
        counts = matches.groupby("회차")["적중"].sum()
        best = tops.groupby("회차")["일치 경기수"].max()
        lines += ["## 확보된 회차의 성적", "", f"분석 경기 수: {len(matches)}", f"단식 적중: {int(matches['적중'].sum())}/{len(matches)} ({matches['적중'].mean()*100:.2f}%)",
                  f"회차 평균 적중: {counts.mean():.2f}/14", f"최고/최저 회차: {int(counts.idxmax())}회차 {int(counts.max())}/14 / {int(counts.idxmin())}회차 {int(counts.min())}/14",
                  *(f"단식 {k}개 일치: {int((counts == k).sum())}회차" for k in (14, 13, 12, 11)),
                  f"TOP20 정답 포함: {int((best == 14).sum())}회차", f"TOP20 최고 일치 분포: {best.value_counts().sort_index().to_dict()}",
                  f"Brier Score: {matches['Brier Score'].mean():.6f}", f"Log Loss: {matches['Log Loss'].mean():.6f}",
                  f"홈승 기준선 적중률: {matches['홈승 기준선 적중'].mean()*100:.2f}%", f"과거 빈도 기준선 적중률: {matches['빈도 기준선 적중'].mean()*100:.2f}%",
                  "과거 데이터의 수정·가용성은 독립 확인되지 않았고, 실제 수익률은 계산하지 않았다.", ""]
        matches = matches.copy()
        matches["월"] = matches["날짜"].str[:7]
        for col in ("리그", "월", "실제 결과", "AI 선택"):
            grouped = matches.groupby(col).agg(경기=("적중", "size"), 적중=("적중", "sum"))
            lines += [f"### {col}별", "", grouped.to_string(), ""]
    lines += ["## 회차별 현황", "", "회차 | 14경기 확보 | 모델 분석 | 단식 적중 | TOP20 최고 | TOP20 정답 | 사유",
              "--- | --- | --- | --- | --- | --- | ---"]
    for _, row in rounds.iterrows():
        lines.append(" | ".join(str(row[col]) for col in ROUND_REPORT_COLUMNS))
    (output / "backtest_2026_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(rounds: pd.DataFrame | None, history_frame: pd.DataFrame | None, output: Path) -> dict[str, pd.DataFrame]:
    """각 회차를 독립 실행하며 입력 해시별 예측 체크포인트를 재사용한다."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    status = empty_round_status()
    match_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if rounds is not None and history_frame is not None:
        _required(rounds, ROUND_COLUMNS, "회차")
        history = load_history(history_frame)
        # 결과는 키에서 제외해도 평가 전에 다시 결합하므로, 점수 변경이 예측에 영향을 주지 않는다.
        history_digest = hashlib.sha256(history_frame.to_json(orient="records", date_format="iso").encode()).hexdigest()
        for n in range(1, 64):
            try:
                prepared = prepare_round(rounds, n)
                fixture_json = prepared.fixtures.to_json(orient="records", date_format="iso")
                key = hashlib.sha256((fixture_json + prepared.cutoff.isoformat() + history_digest + MODEL_VERSION + TEMPORAL_POLICY_VERSION).encode()).hexdigest()
                path = checkpoints / f"round_{n:02d}_{key}.json"
                if path.exists():
                    snapshot = json.loads(path.read_text(encoding="utf-8"))
                else:
                    snapshot = predict(prepared, history)
                    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                summary, matches, tops = evaluate(snapshot, prepared.outcomes, n)
                status.loc[n - 1] = [summary[col] for col in ROUND_REPORT_COLUMNS]
                match_rows.extend(matches)
                top_rows.extend(tops)
            except (ValueError, KeyError, TypeError) as exc:
                message = str(exc)
                status.loc[n - 1, "사유"] = message
                if not message.startswith("14경기 미확보"):
                    status.loc[n - 1, "14경기 확보"] = "확보·검증 필요"
                errors.append({"회차": n, "사유": message})
    else:
        errors = [{"회차": int(row["회차"]), "사유": row["사유"]} for _, row in status.iterrows()]
    match_frame = pd.DataFrame(match_rows, columns=MATCH_REPORT_COLUMNS)
    top_frame = pd.DataFrame(top_rows, columns=TOP_REPORT_COLUMNS)
    error_frame = pd.DataFrame(errors, columns=["회차", "사유"])
    sources = pd.DataFrame([
        {"출처": "베트맨 공식 63회차 결과", "URL": "https://www.betman.co.kr/main/mainPage/gamebuy/winrstDetl.do?gmId=G011&gmTs=260063", "검증": "공개 화면 14경기 결과 및 10월 26일 발표 확인; 데이터 이용·재배포 권한 미확인"},
        {"출처": "샤프픽 회차 목록 (제3자)", "URL": "https://sharppick.app/wdl/", "검증": "63회차 확정·64회차 발매중 표시; 자동 수집·재배포 허가 미확인"},
        {"출처": "샤프픽 이용약관", "URL": "https://sharppick.app/terms/", "검증": "무단 대량 수집·재배포 제한"},
        {"출처": "킬토토 이용약관", "URL": "https://killtoto.com/terms", "검증": "무단 대량 크롤링 제한"},
        {"출처": "더팩트 1회차 기사", "URL": "https://news.tf.co.kr/read/toto/2278987.htm", "검증": "회차 개요 참고; 전체 경기 데이터로 간주하지 않음"},
    ])
    if rounds is not None and "source_url" in rounds:
        for url in rounds["source_url"].dropna().astype(str).unique():
            sources.loc[len(sources)] = {"출처": "사용자 제공 회차 자료", "URL": url, "검증": "사용자 제공; 독립 검증 필요"}
    if history_frame is not None and "source_url" in history_frame:
        for url in history_frame["source_url"].dropna().astype(str).unique():
            sources.loc[len(sources)] = {"출처": "사용자 제공 경기 이력", "URL": url, "검증": "사용자 제공; 당시 가용성 확인 필요"}
    result = {"rounds": status, "matches": match_frame, "top20": top_frame, "errors": error_frame, "sources": sources}
    for name, table in result.items():
        table.to_csv(output / f"backtest_2026_{name}.csv", index=False, encoding="utf-8-sig")
    write_summary(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="2026 승무패 수동 데이터 백테스트")
    parser.add_argument("--rounds", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/local_backtest_2026"))
    args = parser.parse_args()
    if bool(args.rounds) != bool(args.history):
        parser.error("--rounds와 --history는 함께 지정하세요.")
    rounds = pd.read_csv(args.rounds, encoding="utf-8-sig") if args.rounds else None
    history = pd.read_csv(args.history, encoding="utf-8-sig") if args.history else None
    result = run(rounds, history, args.output)
    print(f"분석 완료: {(result['rounds']['모델 분석'] == '완료').sum()}회차 / 대상 63회차")


if __name__ == "__main__":
    main()
