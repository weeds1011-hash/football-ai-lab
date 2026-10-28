from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from draw_diagnostics import diagnose, features, train
from draw_research_workflow import checked_history, create_predictions, reveal_results
from football_data import HistoricalMatch
from backtest_2026 import HISTORY_COLUMNS, ROUND_COLUMNS


def example_data():
    start=datetime(2024,8,1,tzinfo=timezone.utc)
    events=[]
    for i in range(80):
        h=i%4+1; a=(i+1)%4+1
        kickoff=start+timedelta(days=7*i)
        if i%4==0: hg,ag=1,1
        elif i%3==0: hg,ag=0,2
        else: hg,ag=2,1
        m=HistoricalMatch(str(i),kickoff,h,f'Team {h}',a,f'Team {a}',hg,ag)
        events.append((m,pd.Timestamp(kickoff+timedelta(hours=3))))
    cutoff=pd.Timestamp('2025-12-25T00:00:00Z')
    fixture={'number':1,'competition':'PL','kickoff':'2025-12-26T00:00:00Z','home_team':'Team 1','away_team':'Team 2','home_id':1,'away_id':2}
    return {'PL':events},fixture,cutoff


def test_features_and_calibration_outputs_are_valid():
    history,fixture,cutoff=example_data()
    model=train(history,pd.Timestamp('2026-01-01T00:00:00+09:00'))
    d=diagnose(fixture,history,cutoff,model)
    assert len(d['vector'])==8 and d['home_recent5']['n']==5 and d['home_recent10']['n']==10
    assert np.isclose(sum(d['base_probs']),1) and np.isclose(sum(d['experimental_probs']),1)
    assert all(0<=x<=1 for x in d['experimental_probs'])
    assert d['candidate']==(d['risk_score']>=model.threshold)
    assert d['p00']+d['p11']+d['p22']<=d['base_probs'][1]+1e-12


@pytest.mark.parametrize('field',['투표율','배당률','home_odds','vote_share','market_probability','prediction_site_pick','actual_score'])
def test_predictor_rejects_market_or_answer_fields(field):
    history,fixture,cutoff=example_data()
    with pytest.raises(ValueError,match='식별 필드'):
        features({**fixture,field:0.5},history,cutoff)


def test_upload_rejects_market_columns_before_processing():
    rounds=pd.DataFrame(columns=ROUND_COLUMNS+['betting_ratio'])
    with pytest.raises(ValueError,match='시장 정보'):
        create_predictions(rounds,pd.DataFrame(),pd.DataFrame(),1)
    with pytest.raises(ValueError,match='시장 정보'):
        checked_history(pd.DataFrame(columns=HISTORY_COLUMNS+['odds']))


def test_future_result_and_late_confirmation_do_not_change_features_or_model():
    history,fixture,cutoff=example_data()
    old=features(fixture,history,cutoff)
    base=train(history,pd.Timestamp('2026-01-01T00:00:00+09:00'))
    injected=HistoricalMatch('future',datetime(2025,12,24,23,tzinfo=timezone.utc),1,'Team 1',2,'Team 2',50,0)
    altered={'PL':history['PL']+[(injected,pd.Timestamp(cutoff.to_pydatetime()+timedelta(days=10)))]}
    new=features(fixture,altered,cutoff)
    other=train(altered,pd.Timestamp('2026-01-01T00:00:00+09:00'))
    np.testing.assert_allclose(old['vector'],new['vector'])
    np.testing.assert_allclose(base.coefficients,other.coefficients)
    assert base.training_count==other.training_count


def test_changing_cutoff_limits_inputs_and_2026_training_is_rejected():
    history,fixture,cutoff=example_data()
    later=features(fixture,history,cutoff)
    earlier=features(fixture,history,pd.Timestamp('2025-09-01T00:00:00Z'))
    assert earlier['history_count']<later['history_count']
    with pytest.raises(ValueError,match='1월 이전'):
        train(history,pd.Timestamp('2026-07-28T00:00:00Z'))


def test_reveal_refuses_changed_saved_prediction():
    answers=pd.DataFrame([{'number':i,'home_team':'a','away_team':'b','home_score':0,'away_score':0,'outcome':'무'} for i in range(1,15)])
    with pytest.raises(ValueError,match='변경'):
        reveal_results({'predictions':[{}]*14,'prediction_hash':'tampered'},answers)


def test_history_cannot_be_known_before_it_started():
    history,fixture,cutoff=example_data()
    m=history['PL'][0][0]
    history['PL'][0]=(m,pd.Timestamp(m.kickoff-timedelta(hours=1)))
    with pytest.raises(ValueError,match='확인시각'):
        features(fixture,history,cutoff)


def test_full_upload_workflow_separates_predictions_and_results():
    history,_,_=example_data()
    rows=[]
    for m,known in history['PL']:
        rows.append(dict(competition='PL',kickoff=m.kickoff.isoformat(),finished_at=known.isoformat(),
                         home_team=m.home_team,away_team=m.away_team,home_id=m.home_team_id,away_id=m.away_team_id,
                         home_score=m.home_goals,away_score=m.away_goals,source_url='https://example.test',acquired_at=known.isoformat()))
    history_frame=pd.DataFrame(rows,columns=HISTORY_COLUMNS)
    rounds=[]
    for i in range(14):
        home=i%4+1; away=(i+1)%4+1
        rounds.append(dict(year=2026,round=1,number=i+1,competition='PL',kickoff='2026-01-03T12:00:00Z',
            deadline='2026-01-03T11:00:00Z',home_team=f'Team {home}',away_team=f'Team {away}',home_id=home,away_id=away,
            result_status='정상',home_score=1,away_score=1,outcome='무',source_url='https://example.test',verification='확인'))
    payload,answers=create_predictions(pd.DataFrame(rounds,columns=ROUND_COLUMNS),history_frame,history_frame,1)
    assert len(payload['predictions'])==14 and '실제' not in payload['predictions'][0]
    details,summary=reveal_results(payload,answers)
    assert len(details)==14 and set(summary['모델'])=={'기존','보강','진단','후보'}
    assert set(details['실제'])=={'무'}
