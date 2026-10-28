from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from models import ModelConfig
from tactical_experiment import PROFILE_COLUMNS, compare_scenario, exact_match, load_profiles


def profile(name='Team A', ko='에이팀', **changes):
    row=dict(league='EPL',team_name_ko=ko,team_name_en=name,tier=2,squad_ovr=80,
             squad_depth=7.0,crack_factor=8.0,set_piece_threat=7.0,base_formation='4-3-3',
             build_up_style='Hybrid',def_line_height='Mid',ppda_intensity=10.0,rest_defense=7.0,
             transition_speed=8.0,xg_for_90=1.5,xg_against_90=1.2,home_adv_weight=1.2,
             late_stamina=1.0,draw_tendency=0.3,tactical_nemesis='Low_Block')
    return {**row,**changes}


def fixture():
    return {'경기번호':'1','홈팀':'Team A','원정팀':'Team B','홈 기대득점':1.4,'원정 기대득점':1.1,
            '홈 Elo':1510,'원정 Elo':1500}


def test_data_validation_and_exact_team_matching():
    p=load_profiles(pd.DataFrame([profile(),profile('Team B','비팀')],columns=PROFILE_COLUMNS))
    assert exact_match(p,'EPL','team a')==[0]
    assert exact_match(p,'EPL','Team')==[]
    assert exact_match(p,'SERIEA','Team A')==[]
    with pytest.raises(ValueError,match='중복'):
        load_profiles(pd.DataFrame([profile(),profile()]))
    with pytest.raises(ValueError,match='추가 열'):
        load_profiles(pd.DataFrame([{**profile(),'odds':2}]))
    with pytest.raises(ValueError,match='draw_tendency'):
        load_profiles(pd.DataFrame([profile(draw_tendency=0.9)]))


def test_comparison_is_opt_in_normalized_and_keeps_original_inputs():
    match=fixture(); home=profile(); away=profile('Team B','비팀',def_line_height='Low')
    saved=deepcopy((match,home,away))
    with pytest.raises(ValueError,match='직접 확인'):
        compare_scenario(match,home,away,confirmed=False)
    result=compare_scenario(match,home,away,confirmed=True)
    assert np.isclose(sum(result.baseline),1) and np.isclose(sum(result.experimental),1)
    assert all(0<=x<=1 for x in result.experimental)
    assert result.rho_capped and result.safe_rho==pytest.approx(-0.2)
    assert result.home_multiplier==pytest.approx(0.85+0.015*7.5)
    assert (match,home,away)==saved


def test_market_odds_do_not_change_any_simulation_probabilities():
    home=profile();away=profile('Team B','비팀')
    a=compare_scenario(fixture(),home,away,confirmed=True)
    b=compare_scenario({**fixture(),'홈승 배당':5.0,'무승부 배당':2.2,'원정승 배당':1.5,'vote_share':99},home,away,confirmed=True)
    assert a.baseline==b.baseline and a.experimental==b.experimental


def test_invalid_matchup_is_rejected():
    home=profile();away=profile('Team B','비팀',league='MLS')
    with pytest.raises(ValueError,match='같은 리그'):
        compare_scenario(fixture(),home,away,confirmed=True)
    with pytest.raises(ValueError,match='중복'):
        compare_scenario(fixture(),home,home,confirmed=True)


def test_extreme_rho_preserves_valid_score_matrix():
    home=profile(draw_tendency=0.5);away=profile('Team B','비팀',draw_tendency=0.5)
    match={**fixture(),'홈 기대득점':8,'원정 기대득점':8}
    result=compare_scenario(match,home,away,confirmed=True,config=ModelConfig())
    assert result.rho_capped and -0.2<=result.safe_rho<0
    assert all(0<=x<=1 for x in result.experimental)
    assert sum(result.experimental)==pytest.approx(1)
