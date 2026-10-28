"""경기 전 이력만 받는 실험용 무승부 진단. 시장 정보는 허용하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from football_data import HistoricalMatch, _team_events, calculate_elo_ratings, calculate_team_form, estimate_expected_goals
from models import ModelConfig, OUTCOMES, elo_adjusted_goals, score_probability_matrix

FIXTURE_FIELDS = frozenset({'number', 'competition', 'kickoff', 'home_team', 'away_team', 'home_id', 'away_id'})
FEATURE_NAMES = ('elo_gap', 'points5_gap', 'points10_gap', 'expected_total', 'expected_gap',
                 'recent_draw_rate', 'league_draw_rate', 'baseline_draw')
RESEARCH_NOTICE = 'Football AI Lab의 승·무·패 예측은 베트맨 사전 투표율, 베팅 비율 및 배당률을 참고하지 않습니다. 실제 축구 경기 이력과 자체 Elo·Poisson·Dixon–Coles 모델 및 검증된 통계 지표만으로 독립적으로 계산합니다.'


def _stamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('시간대가 있는 기준시각이 필요합니다.')
    return result


def check_fixture(fixture: Mapping[str, object]) -> None:
    if set(fixture) != FIXTURE_FIELDS:
        raise ValueError('경기 식별 필드만 허용합니다. 투표율·배당·정답 및 미확인 추가 열은 거부합니다.')
    if int(fixture['home_id']) == int(fixture['away_id']):
        raise ValueError('홈팀과 원정팀이 같습니다.')


def check_history(history: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]]) -> None:
    for records in history.values():
        for item in records:
            if not isinstance(item, tuple) or len(item) != 2 or type(item[0]) is not HistoricalMatch:
                raise ValueError('허용된 경기 이력 형식만 입력할 수 있습니다.')
            if _stamp(item[1]) <= _stamp(item[0].kickoff):
                raise ValueError('경기 결과 확인시각은 경기 시작보다 늦어야 합니다.')


def _record(events: list[tuple[str, int, int]]) -> dict[str, float | int]:
    n = len(events)
    if not n:
        raise ValueError('경기 전 팀 이력이 없습니다.')
    return {'n': n, 'points': sum(3 if x[1] > x[2] else 1 if x[1] == x[2] else 0 for x in events)/n,
            'draw': sum(x[1] == x[2] for x in events)/n,
            'goals_for': sum(x[1] for x in events)/n, 'goals_against': sum(x[2] for x in events)/n,
            'total_goals': sum(x[1]+x[2] for x in events)/n}


def features(fixture: Mapping[str, object], history: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]], cutoff: object) -> dict:
    """타깃 정답과 시장 필드를 받을 수 없는 엄격한 경기 식별 스키마."""
    check_fixture(fixture)
    check_history(history)
    when = _stamp(cutoff)
    if _stamp(fixture['kickoff']) < when:
        raise ValueError('대상 경기가 기준시각 이전에 시작했습니다.')
    prior = [m for m, known in history.get(str(fixture['competition']).upper(), [])
             if _stamp(known) < when and _stamp(m.kickoff) < when]
    hid, aid = int(fixture['home_id']), int(fixture['away_id'])
    def names_for(team_id: int) -> set[str]:
        return {name for m in prior for id_, name in ((m.home_team_id,m.home_team),(m.away_team_id,m.away_team)) if id_ == team_id}
    if fixture['home_team'] not in names_for(hid) or fixture['away_team'] not in names_for(aid):
        raise ValueError('팀명·팀 ID 불일치 또는 경기 전 자료 부족')
    for team in (fixture['home_team'], fixture['away_team']):
        ids = {id_ for m in prior for id_, name in ((m.home_team_id,m.home_team),(m.away_team_id,m.away_team)) if name == team}
        if len(ids) != 1: raise ValueError('팀명에 여러 ID가 연결되어 있습니다.')
    ratings = calculate_elo_ratings(prior)
    he, ae = _team_events(prior,hid), _team_events(prior,aid)
    h5, a5, h10, a10 = _record(he[-5:]), _record(ae[-5:]), _record(he[-10:]), _record(ae[-10:])
    hf=calculate_team_form(prior,team_id=hid,team_name=str(fixture['home_team']),elo=ratings[hid],prediction_venue='home')
    af=calculate_team_form(prior,team_id=aid,team_name=str(fixture['away_team']),elo=ratings[aid],prediction_venue='away')
    hg,ag=estimate_expected_goals(prior,home_id=hid,away_id=aid)
    ah,aa=elo_adjusted_goals(hg,ag,ratings[hid],ratings[aid])
    matrix=score_probability_matrix(ah,aa)
    probs=[float(np.tril(matrix,-1).sum()),float(np.trace(matrix)),float(np.triu(matrix,1).sum())]
    league_draw=sum(m.home_goals==m.away_goals for m in prior)/len(prior)
    recent_draw=(h5['draw']*h5['n']+a5['draw']*a5['n'])/(h5['n']+a5['n'])
    return {'number':int(fixture['number']), 'competition':str(fixture['competition']),
            'home_team':fixture['home_team'],'away_team':fixture['away_team'],
            'history_count':len(prior),'home_count':len(he),'away_count':len(ae),
            'home_elo':ratings[hid],'away_elo':ratings[aid], 'home_recent5':h5,'away_recent5':a5,
            'home_recent10':h10,'away_recent10':a10,'home_venue':hf.home_record,'away_venue':af.away_record,
            'home_xg':hg,'away_xg':ag,'adjusted_home_xg':ah,'adjusted_away_xg':aa,
            'p00':float(matrix[0,0]),'p11':float(matrix[1,1]),'p22':float(matrix[2,2]),
            'base_probs':probs,'uncertain':min(len(he),len(ae))<5 or min(hf.venue_matches,af.venue_matches)<3,
            'vector':[abs(ratings[hid]-ratings[aid]),abs(h5['points']-a5['points']),abs(h10['points']-a10['points']),
                      ah+aa,abs(ah-aa),recent_draw,league_draw,probs[1]],
            'low_goal_trend':(h5['total_goals']+a5['total_goals'])/2}


@dataclass(frozen=True)
class RiskModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    threshold: float
    training_count: int
    training_draws: int
    training_start: str
    training_end: str
    medians: np.ndarray

    def probability(self, vector: list[float]) -> float:
        z=np.r_[1.0,(np.asarray(vector,dtype=float)-self.mean)/self.scale]
        return float(1/(1+np.exp(-np.clip(z@self.coefficients,-25,25))))


def train(history: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]], training_end: object) -> RiskModel:
    """학습 경기는 2025년 이전에 시작하고, 결과가 종료 전 확인된 경기만 사용한다."""
    check_history(history)
    end=_stamp(training_end)
    if end > pd.Timestamp('2026-01-01T00:00:00+09:00'):
        raise ValueError('이 실험의 학습 종료는 2026년 1월 이전이어야 합니다.')
    items=[]
    for code, records in history.items():
        for match, known in records:
            if _stamp(match.kickoff) < end and _stamp(known) < end:
                fixture={'number':1,'competition':code,'kickoff':match.kickoff,'home_team':match.home_team,
                         'away_team':match.away_team,'home_id':match.home_team_id,'away_id':match.away_team_id}
                try: f=features(fixture,history,match.kickoff)
                except ValueError: continue
                if min(f['home_count'],f['away_count'])<3: continue
                items.append((match.kickoff,f['vector'],int(match.home_goals==match.away_goals)))
    items.sort(key=lambda item:item[0])
    if len(items)<15 or sum(item[2] for item in items)<3 or len(items)-sum(item[2] for item in items)<3:
        raise ValueError('2025년까지 시점 검증된 학습 경기가 부족하여 확률 보정을 실행할 수 없습니다.')
    x=np.asarray([i[1] for i in items],dtype=float); y=np.asarray([i[2] for i in items],dtype=float)
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    beta=np.zeros(design.shape[1]); penalty=np.diag([0.0]+[10.0]*len(FEATURE_NAMES))
    for _ in range(50):
        p=1/(1+np.exp(-np.clip(design@beta,-25,25)))
        gradient=design.T@(p-y)+penalty@beta
        hessian=design.T@((p*(1-p))[:,None]*design)+penalty+np.eye(len(beta))*1e-8
        step=np.linalg.solve(hessian,gradient); beta-=step
        if np.linalg.norm(step)<1e-9: break
    scores=1/(1+np.exp(-np.clip(design@beta,-25,25)))
    # 후보 표시는 학습 점수의 상위 사분위, 미리 정한 탐색 규칙이다.
    return RiskModel(mean,scale,beta,float(np.quantile(scores,0.75)),len(items),int(y.sum()),
                     items[0][0].isoformat(),end.isoformat(),np.median(x,axis=0))


def diagnose(fixture: Mapping[str, object], history: dict[str, list[tuple[HistoricalMatch, pd.Timestamp]]], cutoff: object, model: RiskModel) -> dict:
    f=features(fixture,history,cutoff)
    risk=model.probability(f['vector'])
    base=f['base_probs']; remaining=base[0]+base[2]
    probs=[(1-risk)*base[0]/remaining,risk,(1-risk)*base[2]/remaining]
    reasons=[]
    if f['vector'][0]<=model.medians[0]: reasons.append('Elo 차이 작음')
    if f['vector'][1]<=model.medians[1]: reasons.append('최근 5경기 승점 차이 작음')
    if f['vector'][3]<=model.medians[3]: reasons.append('모델 예상 총득점 낮음')
    if f['vector'][5]>=model.medians[5]: reasons.append('최근 무승부 비율 높음')
    return {**f,'risk_score':risk,'candidate':risk>=model.threshold,'candidate_reasons':', '.join(reasons) or '학습 구간 중앙값 기준 뚜렷한 신호 없음',
            'experimental_probs':probs,'experimental_pick':OUTCOMES[int(np.argmax(probs))],
            'base_pick':OUTCOMES[int(np.argmax(base))]}
