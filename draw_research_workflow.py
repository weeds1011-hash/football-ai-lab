"""무승부 실험의 입력 검증, 예측 고정과 정답 결합을 분리한다."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from backtest_2026 import HISTORY_COLUMNS, ROUND_COLUMNS, Prepared, _time, load_history, prepare_round, predict
from draw_diagnostics import diagnose, train
from models import OUTCOMES


def _exact_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    if frame.columns.duplicated().any() or set(frame.columns) != set(expected):
        raise ValueError(f'{label}: 지정된 열만 허용합니다. 투표율·배당·시장 정보 또는 미확인 추가 열은 거부합니다.')


def checked_history(frame: pd.DataFrame):
    _exact_columns(frame,HISTORY_COLUMNS,'경기 이력')
    frame=frame.reset_index(drop=True)
    groups=load_history(frame)
    # 당시 확인 근거 시각과 경기 종료시각 중 더 늦은 시각을 사용한다.
    return {code:[(m,max(finish,_time(frame.iloc[int(m.match_id)]['acquired_at']))) for m,finish in records]
            for code,records in groups.items()}


def create_predictions(rounds: pd.DataFrame, original_history: pd.DataFrame,
                       enriched_history: pd.DataFrame, round_number: int) -> tuple[dict, pd.DataFrame]:
    _exact_columns(rounds,ROUND_COLUMNS,'회차 자료')
    original=checked_history(original_history); enriched=checked_history(enriched_history)
    prepared=prepare_round(rounds,round_number)
    # 정답 데이터가 포함된 Prepared를 예측 함수에 전달하지 않는다.
    safe=Prepared(prepared.fixtures.copy(),pd.DataFrame(),prepared.cutoff,prepared.basis)
    a=predict(safe,original); b=predict(safe,enriched)
    model=train(enriched,pd.Timestamp('2026-01-01T00:00:00+09:00'))
    rows=[]
    for fixture,pa,pb in zip(safe.fixtures.to_dict('records'),a['predictions'],b['predictions'],strict=True):
        d=diagnose(fixture,enriched,safe.cutoff,model)
        if not np.allclose(d['base_probs'],pb['probabilities'],atol=1e-12,rtol=0):
            raise ValueError('보강 모델의 예측 확률 불일치')
        rows.append({'회차':round_number,'번호':int(fixture['number']),'홈팀':fixture['home_team'],'원정팀':fixture['away_team'],
                     '기존 확률':pa['probabilities'],'기존 선택':pa['pick'],
                     '보강 확률':pb['probabilities'],'보강 선택':pb['pick'],
                     '진단 확률':d['experimental_probs'],'진단 선택':d['experimental_pick'],
                     '후보 확률':pb['probabilities'],'후보 선택':pb['pick'],
                     '후보':bool(d['candidate']),'위험 점수':d['risk_score'],'후보 근거':d['candidate_reasons'],
                     '홈 이력수':d['home_count'],'원정 이력수':d['away_count'],'불확실':bool(d['uncertain']),
                     '홈 Elo':d['home_elo'],'원정 Elo':d['away_elo'],
                     '홈 최근5':d['home_recent5'],'원정 최근5':d['away_recent5'],
                     '홈 최근10':d['home_recent10'],'원정 최근10':d['away_recent10'],
                     '홈 최근 홈성적':d['home_venue'],'원정 최근 원정성적':d['away_venue'],
                     '홈 기대득점':d['home_xg'],'원정 기대득점':d['away_xg'],
                     '0:0 확률':d['p00'],'1:1 확률':d['p11'],'2:2 확률':d['p22'],
                     '리그 무승부 빈도':d['vector'][6]})
    payload={'cutoff':safe.cutoff.isoformat(),'cutoff_basis':safe.basis,'training_count':model.training_count,
             'training_draws':model.training_draws,'training_end':model.training_end,'candidate_threshold':model.threshold,
             'predictions':rows}
    payload['prediction_hash']=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return payload,prepared.outcomes.copy()


def reveal_results(payload: dict, answers: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    if len(payload['predictions'])!=14 or len(answers)!=14: raise ValueError('14경기 예측과 정답이 필요합니다.')
    stable={k:v for k,v in payload.items() if k!='prediction_hash'}
    if hashlib.sha256(json.dumps(stable,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=payload['prediction_hash']:
        raise ValueError('저장한 예측이 변경되었습니다.')
    detail=[]; all_scores=[]
    for p,answer in zip(payload['predictions'],answers.sort_values('number').to_dict('records'),strict=True):
        if (p['번호'],p['홈팀'],p['원정팀']) != (answer['number'],answer['home_team'],answer['away_team']):
            raise ValueError('예측과 정답의 번호 또는 팀이 일치하지 않습니다.')
        truth=answer['outcome']; idx=OUTCOMES.index(truth)
        row={k:v for k,v in p.items() if not isinstance(v,(dict,list))}
        row.update({'실제':truth,'실제 스코어':f"{int(answer['home_score'])}:{int(answer['away_score'])}"})
        for model in ('기존','보강','진단','후보'):
            probs=p[f'{model} 확률']; pick=p[f'{model} 선택']
            if len(probs)!=3 or not np.isclose(sum(probs),1): raise ValueError('확률 오류')
            for label,value in zip(('승%','무%','패%'),probs,strict=True): row[f'{model} {label}']=value*100
            all_scores.append({'모델':model,'실제':truth,'선택':pick,'적중':pick==truth,
                'Brier':sum((v-(i==idx))**2 for i,v in enumerate(probs)),
                'Log Loss':-np.log(max(probs[idx],1e-15)),'후보':p['후보']})
        detail.append(row)
    scores=pd.DataFrame(all_scores); summary=[]
    for model,g in scores.groupby('모델'):
        draws=g[g['실제']=='무']; candidates=g[g['후보']]
        summary.append({'모델':model,'경기수':len(g),'적중':int(g['적중'].sum()),
                        '적중률':g['적중'].mean(),'무승부 적중':int(draws['적중'].sum()),'무승부 수':len(draws),
                        'Brier':g['Brier'].mean(),'Log Loss':g['Log Loss'].mean(),
                        '후보 발견율':(candidates['실제']=='무').sum()/len(draws) if model in ('진단','후보') and len(draws) else None,
                        '후보 정밀도':(candidates['실제']=='무').mean() if model in ('진단','후보') and len(candidates) else None})
    return pd.DataFrame(detail),pd.DataFrame(summary)
