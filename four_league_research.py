"""경기 결과 기술 통계와 사전에 정한 무승부 선택 규칙.

예측 함수는 평가용 결과를 받지 않는다. C의 보정 모델은 별도 시점 검증된
충분한 학습 기록이 없으면 만들지 않는다.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from models import OUTCOMES
from draw_diagnostics import diagnose, train

RELATIVE_DRAW_THRESHOLD=0.90
ELO_GAP_LIMIT=50.0
RULE_DESCRIPTION='무승부 확률이 최고 확률의 90% 이상이고 양 팀 Elo 차이가 50점 이하'


def train_before_evaluation(history, training_end: object, evaluation_start: object):
    """학습 종료는 평가 시작 전이며, 표본이 적으면 실험 보정을 거부한다."""
    end=pd.Timestamp(training_end); start=pd.Timestamp(evaluation_start)
    if end.tzinfo is None or start.tzinfo is None or end>=start:
        raise ValueError('학습 종료는 평가 시작보다 앞선 시간이어야 합니다.')
    model=train(history,end,latest_training_end=start)
    if model.training_count<100 or model.training_draws<20:
        raise ValueError('독립 평가에 사용할 경기 전 학습 표본이 부족합니다 (최소 100경기·무승부 20경기).')
    return model


def calibrated_prediction(fixture,history,cutoff,model):
    """보정 모델 C를 평가하며 원래 기본 확률을 함께 보존한다."""
    return diagnose(fixture,history,cutoff,model)


def frozen_decisions(probabilities: list[float], elo_gap: float) -> dict[str, object]:
    probs=np.asarray(probabilities,dtype=float)
    if probs.shape!=(3,) or not np.isfinite(probs).all() or (probs<0).any() or not np.isclose(probs.sum(),1):
        raise ValueError('승·무·패 확률이 올바르지 않습니다.')
    if not math.isfinite(elo_gap) or elo_gap<0: raise ValueError('Elo 차이가 잘못됐습니다.')
    base=OUTCOMES[int(np.argmax(probs))]
    candidate=bool(probs[1]>=RELATIVE_DRAW_THRESHOLD*float(probs.max()) and elo_gap<=ELO_GAP_LIMIT)
    return {'A 선택':base,'B 선택':base,'B 후보':candidate,
            'D 선택':'무' if candidate else base,'후보 근거':RULE_DESCRIPTION if candidate else '사전 설정한 두 조건을 모두 충족하지 않음'}


def descriptive_results(rows: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    """결과 통계이며 예측에 사용되지 않는다. 종료 확인은 입력 시각의 사실만 뜻한다."""
    required={'league','kickoff','status','home_score','away_score'}
    if not required.issubset(rows.columns): raise ValueError('경기 결과 필드가 부족합니다.')
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or end<=start: raise ValueError('시간대 포함 분석 구간이 필요합니다.')
    df=rows.copy();df['kickoff']=pd.to_datetime(df['kickoff'],utc=True,errors='coerce')
    df=df[(df['kickoff']>=start)&(df['kickoff']<end)&(df['status']=='FINISHED')].copy()
    for col in ('home_score','away_score'): df[col]=pd.to_numeric(df[col],errors='coerce')
    if df[['home_score','away_score']].isna().any().any() or (df[['home_score','away_score']]<0).any().any():
        raise ValueError('완료 경기의 스코어가 누락됐습니다.')
    df['draw']=df['home_score']==df['away_score']
    result=[]
    for league,g in df.groupby('league',sort=True):
        draws=g[g['draw']]
        result.append({'리그':league,'경기수':len(g),'무승부':len(draws),'무승부율':len(draws)/len(g),
            '0:0':int((draws['home_score']==0).sum()),'1:1':int((draws['home_score']==1).sum()),
            '2:2 이상':int((draws['home_score']>=2).sum())})
    return pd.DataFrame(result,columns=['리그','경기수','무승부','무승부율','0:0','1:1','2:2 이상'])


def evaluate_fixed(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    required={'id','league','kickoff','home_team','away_team','probs','A 선택','B 후보','D 선택'}
    if not required.issubset(predictions.columns) or not {'id','home_team','away_team','actual'}.issubset(outcomes.columns):
        raise ValueError('예측 또는 결과 필드가 부족합니다.')
    if predictions['id'].duplicated().any() or outcomes['id'].duplicated().any(): raise ValueError('중복 경기 ID')
    matched=predictions.merge(outcomes,on='id',validate='one_to_one',suffixes=('','_actual'))
    if len(matched)!=len(predictions) or (matched.home_team!=matched.home_team_actual).any() or (matched.away_team!=matched.away_team_actual).any():
        raise ValueError('예측과 정답의 경기 또는 홈원정이 일치하지 않습니다.')
    rows=[]
    for p in matched.to_dict('records'):
        if p['actual'] not in OUTCOMES: raise ValueError('정답 값이 올바르지 않습니다.')
        actual_i=OUTCOMES.index(p['actual']);pr=p['probs']
        for label in ('A','B','C','D') if 'C 확률' in p else ('A','B','D'):
            candidate_probs=p['C 확률'] if label=='C' else pr
            if len(candidate_probs)!=3 or not np.isclose(sum(candidate_probs),1): raise ValueError('보정 확률 오류')
            choice=(p['A 선택'] if label in ('A','B') else p['D 선택'] if label=='D'
                    else OUTCOMES[int(np.argmax(candidate_probs))])
            rows.append({'모델':label,'id':p['id'],'리그':p['league'],'날짜':p['kickoff'],
                '실제':p['actual'],'선택':choice,'적중':choice==p['actual'],'후보':p['B 후보'],
                'Brier':sum((q-(i==actual_i))**2 for i,q in enumerate(candidate_probs)),
                'Log Loss':-math.log(max(candidate_probs[actual_i],1e-15))})
    detail=pd.DataFrame(rows); summary=[]
    for label,g in detail.groupby('모델'):
        is_draw=g['실제']=='무'; predicted_draw=g['선택']=='무'
        summary.append({'모델':label,'경기수':len(g),'적중':int(g['적중'].sum()),'적중률':g['적중'].mean(),
            '실제 무승부':int(is_draw.sum()),'무승부 선택':int(predicted_draw.sum()),
            '무승부 적중':int((is_draw&predicted_draw).sum()),
            '무승부 정밀도':float((is_draw&predicted_draw).sum()/predicted_draw.sum()) if predicted_draw.sum() else None,
            '무승부 재현율':float((is_draw&predicted_draw).sum()/is_draw.sum()) if is_draw.sum() else None,
            '후보 수':int(g['후보'].sum()) if label=='B' else None,
            '후보 실제 무승부':int((g['후보']&is_draw).sum()) if label=='B' else None,
            'Brier':g['Brier'].mean(),'Log Loss':g['Log Loss'].mean()})
    return detail,pd.DataFrame(summary)
