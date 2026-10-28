import pandas as pd
import pytest

from four_league_research import descriptive_results, evaluate_fixed, frozen_decisions, train_before_evaluation


def test_descriptive_draw_counts_and_date_bounds():
    rows=pd.DataFrame([
        {'league':'PL','kickoff':'2026-08-01T00:00:00Z','status':'FINISHED','home_score':0,'away_score':0},
        {'league':'PL','kickoff':'2026-08-02T00:00:00Z','status':'FINISHED','home_score':2,'away_score':2},
        {'league':'PL','kickoff':'2026-08-03T00:00:00Z','status':'FINISHED','home_score':2,'away_score':1},
        {'league':'PL','kickoff':'2026-08-04T00:00:00Z','status':'TIMED','home_score':None,'away_score':None},
    ])
    result=descriptive_results(rows,'2026-07-28T00:00:00Z','2026-08-04T00:00:00Z').iloc[0]
    assert result['경기수']==3 and result['무승부']==2 and result['0:0']==1 and result['2:2 이상']==1


def test_draw_rule_is_fixed_and_probability_does_not_change():
    probabilities=[.37,.34,.29]
    result=frozen_decisions(probabilities,25)
    assert result['A 선택']=='승' and result['B 선택']=='승'
    assert result['B 후보'] and result['D 선택']=='무'
    assert probabilities==[.37,.34,.29]
    assert not frozen_decisions(probabilities,51)['B 후보']
    with pytest.raises(ValueError,match='확률'):
        frozen_decisions([.4,.4,.4],0)


def test_selection_and_candidate_metrics_do_not_change_brier():
    decision=frozen_decisions([.37,.34,.29],25)
    p=pd.DataFrame([{'id':1,'league':'PL','kickoff':'2026-09-01T00:00:00Z',
                     'home_team':'A','away_team':'B','probs':[.37,.34,.29],**decision}])
    actual=pd.DataFrame([{'id':1,'home_team':'A','away_team':'B','actual':'무'}])
    detail,summary=evaluate_fixed(p,actual)
    results=summary.set_index('모델')
    assert results.loc['A','적중']==0 and results.loc['D','적중']==1
    assert results.loc['B','후보 실제 무승부']==1
    assert results['Brier'].nunique()==1 and results['Log Loss'].nunique()==1
    bad=actual.copy();bad['away_team']='other'
    with pytest.raises(ValueError,match='홈원정'):
        evaluate_fixed(p,bad)


def test_calibration_rejects_future_and_inadequate_training():
    from test_draw_diagnostics import example_data
    history,_,_=example_data()
    with pytest.raises(ValueError,match='학습 종료'):
        train_before_evaluation(history,'2026-09-01T00:00:00Z','2026-08-01T00:00:00Z')
    with pytest.raises(ValueError,match='표본이 부족'):
        train_before_evaluation(history,'2026-01-01T00:00:00Z','2026-09-01T00:00:00Z')


def test_calibrated_probability_score_is_kept_separate():
    decision=frozen_decisions([.37,.34,.29],25)
    pred=pd.DataFrame([{'id':1,'league':'PL','kickoff':'2026-09-01T00:00:00Z','home_team':'A','away_team':'B',
                        'probs':[.37,.34,.29],'C 확률':[.25,.50,.25],**decision}])
    actual=pd.DataFrame([{'id':1,'home_team':'A','away_team':'B','actual':'무'}])
    _,s=evaluate_fixed(pred,actual)
    s=s.set_index('모델')
    assert s.loc['C','적중']==1 and s.loc['C','Brier']!=s.loc['A','Brier']
