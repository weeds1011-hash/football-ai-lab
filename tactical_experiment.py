"""사용자 제공 미검증 전술 값을 사용하는 선택형 가상 시뮬레이션.

이 값들은 실측 xG나 검증된 선수 정보가 아니다. 기본 예측과 백테스트를 변경하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Mapping

import numpy as np
import pandas as pd

from models import ModelConfig, analyze_match, elo_adjusted_goals, normalise_match_dataframe

PROFILE_COLUMNS = ('league','team_name_ko','team_name_en','tier','squad_ovr','squad_depth',
    'crack_factor','set_piece_threat','base_formation','build_up_style','def_line_height',
    'ppda_intensity','rest_defense','transition_speed','xg_for_90','xg_against_90',
    'home_adv_weight','late_stamina','draw_tendency','tactical_nemesis')
RANGES = {'tier':(1,4),'squad_ovr':(60,95),'squad_depth':(1,10),'crack_factor':(1,10),
          'set_piece_threat':(1,10),'ppda_intensity':(7,16.5),'rest_defense':(1,10),
          'transition_speed':(1,10),'xg_for_90':(0.7,2.6),'xg_against_90':(0.6,2.4),
          'home_adv_weight':(1.05,1.35),'late_stamina':(0.75,1.15),'draw_tendency':(0.1,0.5)}
LEAGUES = {'EPL','LALIGA','SERIEA','MLS','SUPERLIG','SCOT','J1'}


def load_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """추가 열도 거부해 투표율·배당·예상 등 시장 변수 유입을 막는다."""
    if frame.columns.duplicated().any() or set(frame.columns)!=set(PROFILE_COLUMNS):
        raise ValueError('전술 자료는 지정된 20개 열만 허용합니다. 투표율·배당 등 추가 열은 사용할 수 없습니다.')
    df=frame.loc[:,PROFILE_COLUMNS].copy()
    if df.empty or df.isna().any().any(): raise ValueError('팀 자료가 비었거나 누락된 값이 있습니다.')
    for col in ('league','team_name_ko','team_name_en','base_formation','build_up_style','def_line_height','tactical_nemesis'):
        df[col]=df[col].astype(str).str.strip()
        if df[col].eq('').any(): raise ValueError(f'{col}: 빈 값이 있습니다.')
    if not set(df.league).issubset(LEAGUES): raise ValueError('지원하지 않는 리그 코드가 있습니다.')
    if not set(df.def_line_height).issubset({'High','Mid','Low'}): raise ValueError('수비 라인 코드 오류')
    if not set(df.build_up_style).issubset({'Possession','Hybrid','Direct','Counter'}): raise ValueError('전개 방식 코드 오류')
    if not set(df.tactical_nemesis).issubset({'Low_Block','Fast_Transition','High_Press','Set_Piece','Possession','Crack_Individual'}):
        raise ValueError('전술 유형 코드 오류')
    if not df.base_formation.str.fullmatch(r'\d(?:-\d){1,4}').all(): raise ValueError('전술 대형 형식 오류')
    for col,(lo,hi) in RANGES.items():
        df[col]=pd.to_numeric(df[col],errors='coerce')
        if (~np.isfinite(df[col])).any() or (~df[col].between(lo,hi)).any(): raise ValueError(f'{col}: 허용 범위 밖입니다.')
    if not (df.tier % 1 == 0).all() or not (df.squad_ovr % 1 == 0).all():
        raise ValueError('tier와 squad_ovr은 정수여야 합니다.')
    if df.duplicated(['league','team_name_en']).any() or df.duplicated(['league','team_name_ko']).any():
        raise ValueError('같은 리그에 중복된 팀명이 있습니다.')
    return df.reset_index(drop=True)


def _key(name: object) -> str:
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(name)).casefold()).strip()


def exact_match(profile_frame: pd.DataFrame, league: str, team: str) -> list[int]:
    """팀명만 보고 유사한 다른 팀을 자동 연결하지 않는다."""
    return [int(i) for i,row in profile_frame.iterrows() if row['league']==league and
            _key(team) in {_key(row['team_name_en']),_key(row['team_name_ko'])}]


@dataclass(frozen=True)
class Scenario:
    baseline: tuple[float,float,float]
    experimental: tuple[float,float,float]
    baseline_score: str
    experimental_score: str
    home_multiplier: float
    away_multiplier: float
    raw_rho: float
    safe_rho: float
    rho_capped: bool
    notes: tuple[str,...]


def compare_scenario(match: Mapping[str,object], home: Mapping[str,object], away: Mapping[str,object], *,
                     confirmed: bool, config: ModelConfig | None = None) -> Scenario:
    """선택된 한 경기에만 사용자 가정을 적용한다. 기본 입력과 설정을 변경하지 않는다."""
    if not confirmed: raise ValueError('홈팀·원정팀의 프로필 연결을 직접 확인해야 합니다.')
    df=load_profiles(pd.DataFrame([home,away]))
    h,a=df.iloc[0],df.iloc[1]
    if h.league!=a.league or h.team_name_en==a.team_name_en: raise ValueError('같은 리그의 서로 다른 두 팀을 선택하세요.')
    # 기존 모델의 데이터 유효성 검사를 재사용한다. 배당은 출력에도 전달하지 않는다.
    raw=dict(match)
    for key in ('home_odds','draw_odds','away_odds','홈승 배당','무승부 배당','원정승 배당'):
        raw.pop(key,None)
    normalized=normalise_match_dataframe(pd.DataFrame([raw])).iloc[0].to_dict()
    for key in ('home_odds','draw_odds','away_odds'): normalized[key]=np.nan
    cfg=config or ModelConfig()
    baseline=analyze_match(normalized,cfg)
    home_factor=away_factor=1.0
    notes=[]
    if a.def_line_height=='Low':
        home_factor*=0.85+0.015*(h.crack_factor+h.set_piece_threat)/2
        notes.append('원정 로우 블록 가정')
    if h.def_line_height=='Low':
        away_factor*=0.85+0.015*(a.crack_factor+a.set_piece_threat)/2
        notes.append('홈 로우 블록 가정')
    if a.def_line_height=='High':
        home_factor*=1+0.04*max(0,h.transition_speed-a.rest_defense)
        notes.append('원정 하이 라인과 홈 전환 가정')
    if h.def_line_height=='High':
        away_factor*=1+0.04*max(0,a.transition_speed-h.rest_defense)
        notes.append('홈 하이 라인과 원정 전환 가정')
    stamina=math.sqrt(float(h.late_stamina)/float(a.late_stamina))
    home_factor*=stamina; away_factor/=stamina
    notes.append('사용자 체력 계수 가정')
    raw_rho=-float((h.draw_tendency+a.draw_tendency)/2)*math.exp(-abs(float(h.squad_ovr-a.squad_ovr))/15)
    candidate={**normalized,'home_xg':normalized['home_xg']*home_factor,'away_xg':normalized['away_xg']*away_factor}
    if max(candidate['home_xg'],candidate['away_xg'])>10: raise ValueError('시나리오 기대득점이 모델 허용 범위를 벗어났습니다.')
    adjusted=elo_adjusted_goals(candidate['home_xg'],candidate['away_xg'],candidate['home_elo'],candidate['away_elo'],cfg)
    # 네 저득점 보정항이 음수가 되지 않도록 제한. 제한 전·후 값을 모두 표시한다.
    min_rho=max(-0.2,-0.999999/max(adjusted))
    safe_rho=max(min_rho,min(0.2,raw_rho))
    sim_config=ModelConfig(max_goals=cfg.max_goals,dixon_coles_rho=safe_rho,
                           home_advantage_elo=cfg.home_advantage_elo,elo_goal_scale=cfg.elo_goal_scale)
    simulation=analyze_match(candidate,sim_config)
    return Scenario(baseline=baseline.probabilities,experimental=simulation.probabilities,
        baseline_score=f'{baseline.predicted_home_goals}:{baseline.predicted_away_goals}',
        experimental_score=f'{simulation.predicted_home_goals}:{simulation.predicted_away_goals}',
        home_multiplier=home_factor,away_multiplier=away_factor,raw_rho=raw_rho,safe_rho=safe_rho,
        rho_capped=not math.isclose(raw_rho,safe_rho,abs_tol=1e-12),notes=tuple(notes))
