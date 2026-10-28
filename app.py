"""Football AI Lab Streamlit 애플리케이션."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from football_data import (
    FREE_COMPETITIONS,
    FootballDataClient,
    FootballDataError,
    FootballDataRateLimitError,
    SEOUL_TIMEZONE,
    build_real_match_bundle,
    season_start_year,
)

from models import (
    ModelConfig,
    analyze_dataframe,
    calculate_top_combinations,
    read_match_csv,
    toto_insight_frames,
    validate_toto_14,
)


APP_DIR = Path(__file__).resolve().parent
DEMO_CSV = APP_DIR / "data" / "demo_14_matches.csv"

st.set_page_config(
    page_title="Football AI Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 1.2rem; padding-bottom: 3rem;}
    .stButton > button, .stDownloadButton > button {
        min-height: 3.1rem; width: 100%; border-radius: 0.75rem; font-weight: 700;
    }
    div[data-baseweb="tab-list"] {gap: .25rem;}
    button[data-baseweb="tab"] {min-height: 3rem; padding-left: .8rem; padding-right: .8rem;}
    .data-label {
        display: inline-block; padding: .35rem .65rem; border-radius: 999px;
        background: #fff3cd; color: #664d03; font-weight: 700; margin-bottom: .75rem;
    }
    .real-label {
        display: inline-block; padding: .35rem .65rem; border-radius: 999px;
        background: #d1e7dd; color: #0f5132; font-weight: 700; margin-bottom: .75rem;
    }
    @media (max-width: 640px) {
        .block-container {padding: .75rem .65rem 2rem;}
        h1 {font-size: 1.75rem !important;}
        h2 {font-size: 1.35rem !important;}
        p, li, label {font-size: 1rem !important; line-height: 1.55;}
        input {font-size: 16px !important;}
        div[data-testid="stDataFrame"] {font-size: .92rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV, encoding="utf-8-sig")


@st.cache_data(ttl=900, show_spinner=False)
def load_real_api_payload(
    competition_code: str, season: int, *, _api_key: str
) -> tuple[dict[str, object], dict[str, object]]:
    """같은 리그·시즌 API 응답을 15분간 캐시한다."""

    response = FootballDataClient(_api_key).competition_matches(
        competition_code, season
    )
    return dict(response.data), {
        "remaining": response.rate_limit.remaining_per_minute,
        "reset_seconds": response.rate_limit.reset_seconds,
        "observed_at": datetime.now(SEOUL_TIMEZONE).isoformat(),
    }


def football_data_api_key() -> str | None:
    environment_value = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if environment_value:
        return environment_value
    try:
        secret_value = str(st.secrets.get("FOOTBALL_DATA_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError, StreamlitSecretNotFoundError):
        return None
    return secret_value or None


def real_api_data(prefix: str) -> tuple[pd.DataFrame | None, str]:
    st.markdown('<span class="real-label">실제 일정·결과</span>', unsafe_allow_html=True)
    st.caption(
        "football-data.org 실제 일정·최종 스코어를 사용합니다. 기대득점은 실측 xG가 아니라 "
        "과거 득실점으로 계산한 모델 추정치입니다."
    )
    selected_date = st.date_input(
        "경기 날짜 (한국시간)",
        value=datetime.now(SEOUL_TIMEZONE).date(),
        key=f"{prefix}_api_date",
    )
    competition = st.selectbox(
        "리그",
        FREE_COMPETITIONS,
        format_func=lambda item: item.label,
        key=f"{prefix}_api_competition",
    )
    selected_season = season_start_year(selected_date, competition.code)
    st.caption(f"API 조회 시즌 시작 연도: {selected_season} · 동일 조회는 15분간 캐시됩니다.")

    api_key = football_data_api_key()
    if not api_key:
        st.warning(
            "football-data.org API 키가 설정되지 않았습니다. 로컬 `.streamlit/secrets.toml` "
            "또는 Streamlit Cloud Secrets에 `FOOTBALL_DATA_API_KEY`를 설정하세요. "
            "그 전에는 CSV 업로드를 사용할 수 있습니다."
        )
        return None, "실제 일정·결과 / 모델 추정 기대득점"

    state_key = f"{prefix}_real_api_bundle"
    if st.button("실제 경기 목록 불러오기", type="primary", key=f"{prefix}_api_load"):
        try:
            with st.spinner("실제 경기와 이전 결과를 불러오는 중입니다..."):
                payload, rate_limit = load_real_api_payload(
                    competition.code, selected_season, _api_key=api_key
                )
                bundle = build_real_match_bundle(
                    payload,
                    target_date=selected_date,
                    now=datetime.now(SEOUL_TIMEZONE),
                    observed_at=datetime.fromisoformat(str(rate_limit["observed_at"])),
                )
            st.session_state[state_key] = {
                "date": selected_date.isoformat(),
                "competition_code": competition.code,
                "bundle": bundle,
                "rate_limit": rate_limit,
            }
        except FootballDataRateLimitError as exc:
            retry_message = (
                f" 약 {exc.retry_after_seconds}초 후 재시도할 수 있습니다."
                if exc.retry_after_seconds is not None
                else ""
            )
            st.error(f"{exc}{retry_message} CSV 업로드는 계속 사용할 수 있습니다.")
        except FootballDataError as exc:
            st.error(f"{exc} CSV 업로드는 계속 사용할 수 있습니다.")

    stored = st.session_state.get(state_key)
    if not isinstance(stored, dict):
        st.info("날짜와 리그를 선택한 뒤 실제 경기 목록을 불러오세요.")
        return None, "실제 일정·결과 / 모델 추정 기대득점"
    if (
        stored.get("date") != selected_date.isoformat()
        or stored.get("competition_code") != competition.code
    ):
        st.info("선택값이 바뀌었습니다. 실제 경기 목록을 다시 불러오세요.")
        return None, "실제 일정·결과 / 모델 추정 기대득점"

    bundle = stored["bundle"]
    rate_limit = stored["rate_limit"]
    if bundle.input_frame.empty:
        st.info(
            "선택 날짜에 아직 시작하지 않은 경기가 없습니다. 다른 날짜를 선택하거나 CSV를 사용하세요."
        )
        return None, "실제 일정·결과 / 모델 추정 기대득점"

    st.success(
        f"실제 경기 {len(bundle.input_frame)}개를 준비했습니다. "
        f"예측 시점 이전 완료 경기 {bundle.history_match_count}개만 전력 계산에 사용했습니다."
    )
    st.caption("표시한 경기들은 현재 공통 분석 기준시각의 정보만 사용합니다. API의 종료 상태는 조회시각에 확인했으며 정확한 종료시각은 제공하지 않습니다.")
    if not bundle.evidence_frame.empty and (bundle.evidence_frame['데이터 충분성'] == '낮음').any():
        st.warning("경기 이력이 부족한 팀이 있습니다. 데이터 충분성은 예측 확률이나 적중 보장과 다릅니다.")
    if rate_limit.get("remaining") is not None:
        st.caption(
            f"API 분당 잔여 요청: {rate_limit['remaining']}회 · 누락되어 건너뛴 경기: "
            f"{bundle.skipped_match_count}개"
        )
    with st.expander("팀 전력 및 기대득점 산출 근거", expanded=True):
        st.dataframe(bundle.evidence_frame, width="stretch", hide_index=True)
        st.caption(
            "불확실성은 이용 가능한 전체·홈/원정 경기 수로 표시합니다. "
            "실제 라인업, 부상자, 실측 xG는 포함하지 않습니다."
        )
    return bundle.input_frame, "실제 일정·결과 / 모델 추정 기대득점"


def uploaded_or_demo_data(prefix: str) -> tuple[pd.DataFrame | None, str]:
    source = st.radio(
        "데이터 선택",
        ("14경기 가상 데모", "실제 데이터 API", "CSV 업로드"),
        horizontal=True,
        key=f"{prefix}_source",
    )
    active_source_key = f"{prefix}_active_source"
    if st.session_state.get(active_source_key) != source:
        st.session_state[active_source_key] = source
        st.session_state.pop("proto_result" if prefix == "proto" else "toto_bundle", None)
        st.session_state.pop(f"{prefix}_snapshot", None)
    if source == "14경기 가상 데모":
        st.markdown('<span class="data-label">가상 데이터</span>', unsafe_allow_html=True)
        st.caption("팀·기대득점·Elo·배당은 기능 시험을 위해 만든 값이며 실제 경기가 아닙니다.")
        return load_demo_data(), "가상 데이터"
    if source == "실제 데이터 API":
        return real_api_data(prefix)

    uploaded = st.file_uploader(
        "경기 CSV 파일",
        type=("csv",),
        help="UTF-8 또는 CP949 CSV를 지원합니다. 업로드 자료의 출처와 정확성은 사용자가 확인해야 합니다.",
        key=f"{prefix}_upload",
    )
    if uploaded is None:
        st.info("CSV를 선택하면 분석할 수 있습니다.")
        return None, "사용자 업로드"
    try:
        return read_match_csv(uploaded.getvalue()), "사용자 업로드"
    except ValueError as exc:
        st.error(str(exc))
        return None, "사용자 업로드"


def result_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def make_prediction_snapshot(inputs: pd.DataFrame, results: pd.DataFrame, data_type: str, config: ModelConfig, prefix: str) -> str:
    """배당·비밀키를 제외한 분석 입력과 데이터 기준시각을 기록한다."""
    safe_inputs=inputs.drop(columns=["home_odds", "draw_odds", "away_odds"],errors="ignore")
    safe_results=results.drop(columns=[c for c in results.columns if '배당' in c or '기대수익률' in c],errors="ignore")
    state=st.session_state.get(f"{prefix}_real_api_bundle") if data_type.startswith("실제 일정") else None
    evidence=state['bundle'].evidence_frame if isinstance(state,dict) else pd.DataFrame()
    timing = ('API 조회 당시 종료 상태를 확인; 기록의 정확한 종료시각은 미제공' if not evidence.empty
              else '가상 입력' if data_type=='가상 데이터' else '업로드 입력의 당시 가용시각은 미확인')
    return json.dumps({"생성시각":datetime.now(SEOUL_TIMEZONE).isoformat(),"자료구분":data_type,
        "입력 시점 검증":timing,
        "모델":"Elo + Poisson + Dixon–Coles","설정":{"rho":config.dixon_coles_rho,"홈 이점":config.home_advantage_elo},
        "입력":json.loads(safe_inputs.to_json(orient="records",force_ascii=False)),
        "결과":json.loads(safe_results.to_json(orient="records",force_ascii=False)),
        "데이터 기준 및 충분성":json.loads(evidence.to_json(orient="records",force_ascii=False))},ensure_ascii=False,indent=2)


def show_input_preview(frame: pd.DataFrame) -> None:
    with st.expander("입력 데이터 미리보기", expanded=False):
        st.dataframe(frame, width="stretch", hide_index=True)


def show_probability_table(result: pd.DataFrame) -> None:
    columns = [
        "경기번호",
        "홈팀",
        "원정팀",
        "홈승 확률(%)",
        "무승부 확률(%)",
        "원정승 확률(%)",
        "예상 스코어",
    ]
    st.dataframe(
        result.loc[:, columns],
        width="stretch",
        hide_index=True,
        column_config={
            "홈승 확률(%)": st.column_config.ProgressColumn(
                "홈승 %", min_value=0, max_value=100, format="%.2f"
            ),
            "무승부 확률(%)": st.column_config.ProgressColumn(
                "무 %", min_value=0, max_value=100, format="%.2f"
            ),
            "원정승 확률(%)": st.column_config.ProgressColumn(
                "원정승 %", min_value=0, max_value=100, format="%.2f"
            ),
        },
    )
    st.caption("예상 스코어는 가장 가능성이 큰 단일 점수입니다. 승·무·패는 해당하는 여러 스코어의 확률을 합하므로 최우선 결과가 다를 수 있습니다.")


st.title("⚽ Football AI Lab")
st.caption("포아송 · Dixon–Coles · Elo 기반의 설명 가능한 축구 분석 MVP")
st.warning(
    "분석값은 통계 모델의 추정치이며 정확도나 수익을 보장하지 않습니다. "
    "자동 베팅·구매 기능은 제공하지 않습니다."
)

with st.expander("모델 설정", expanded=False):
    rho = st.slider(
        "Dixon–Coles 저득점 보정값 (rho)",
        min_value=-0.20,
        max_value=0.20,
        value=-0.08,
        step=0.01,
        help="0에 가까우면 독립 포아송과 유사합니다.",
    )
    home_advantage = st.slider(
        "홈 어드밴티지 (Elo 점수)",
        min_value=0,
        max_value=120,
        value=65,
        step=5,
    )

config = ModelConfig(dixon_coles_rho=rho, home_advantage_elo=float(home_advantage))

proto_tab, toto_tab, year_tab, review_tab, tactics_tab = st.tabs(("프로토 분석", "승무패 14경기", "2026 승무패 백테스트", "승무패 과거검증", "전술 가상 실험"))

with proto_tab:
    st.subheader("프로토 경기 분석")
    proto_frame, proto_data_type = uploaded_or_demo_data("proto")
    if proto_frame is not None:
        show_input_preview(proto_frame)
        if st.button("프로토 분석 실행", type="primary", key="proto_run"):
            try:
                proto_inputs, proto_result, _ = analyze_dataframe(
                    proto_frame, config, default_data_type=proto_data_type
                )
                st.session_state["proto_result"] = proto_result
                st.session_state["proto_snapshot"] = make_prediction_snapshot(proto_inputs,proto_result,proto_data_type,config,"proto")
            except ValueError as exc:
                st.error(f"입력 데이터 오류: {exc}")

    proto_result = st.session_state.get("proto_result")
    if proto_result is not None:
        st.success(f"{len(proto_result)}경기 분석을 완료했습니다.")
        show_probability_table(proto_result)
        st.markdown("#### 적정 배당과 기대수익률")
        expected_return_columns = [
            "홈승 기대수익률(%)",
            "무승부 기대수익률(%)",
            "원정승 기대수익률(%)",
        ]
        odds_columns = [
            "경기번호",
            "홈승 적정배당",
            "무승부 적정배당",
            "원정승 적정배당",
        ]
        has_market_odds = proto_result[expected_return_columns].notna().any(axis=None)
        display_columns = odds_columns + (
            expected_return_columns if has_market_odds else []
        ) + ["데이터 구분"]
        st.dataframe(
            proto_result.loc[:, display_columns],
            width="stretch",
            hide_index=True,
        )
        if has_market_odds:
            st.caption(
                "기대수익률 = 모델 확률 × 입력 배당 − 1. "
                "양수라도 실제 수익을 보장하지 않습니다."
            )
        else:
            st.info("실제 배당 입력이 없어 기대수익률 계산을 생략했습니다.")
        st.download_button(
            "분석 결과 CSV 다운로드",
            data=result_csv_bytes(proto_result),
            file_name="football_ai_lab_proto_result.csv",
            mime="text/csv",
            key="proto_download",
        )
        if st.session_state.get("proto_snapshot"):
            st.download_button("예측 스냅샷 다운로드",st.session_state["proto_snapshot"],file_name="proto_snapshot.json",mime="application/json",key="proto_snapshot_download")

with toto_tab:
    st.subheader("축구토토 승무패 14경기")
    toto_frame, toto_data_type = uploaded_or_demo_data("toto")
    if toto_data_type == '사용자 업로드':
        st.warning('업로드된 예상 득점·Elo의 공통 마감 전 이용 가능 여부는 자동 확인할 수 없습니다. 엄격한 회차 시점 검증이 필요한 경우 과거검증의 시각이 포함된 회차·이력 파일을 사용하세요.')
    if toto_frame is not None:
        show_input_preview(toto_frame)
        if st.button("14경기 조합 분석", type="primary", key="toto_run"):
            try:
                toto_inputs, toto_result, toto_analyses = analyze_dataframe(
                    toto_frame, config, default_data_type=toto_data_type
                )
                validate_toto_14(toto_analyses)
                combination_search = calculate_top_combinations(toto_analyses, top_n=20)
                draw_frame, upset_frame = toto_insight_frames(toto_analyses)
                st.session_state["toto_bundle"] = (
                    toto_result,
                    combination_search,
                    draw_frame,
                    upset_frame,
                )
                st.session_state["toto_snapshot"] = make_prediction_snapshot(toto_inputs,toto_result,toto_data_type,config,"toto")
            except ValueError as exc:
                st.error(f"입력 데이터 오류: {exc}")

    toto_bundle = st.session_state.get("toto_bundle")
    if toto_bundle is not None:
        toto_result, combination_search, draw_frame, upset_frame = toto_bundle
        st.success("14경기 분석을 완료했습니다.")
        show_probability_table(toto_result)

        st.markdown("#### 확률 상위 20개 조합")
        st.dataframe(
            combination_search.to_frame(),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"전체 이론 조합 {combination_search.total_space:,}개를 메모리에 만들지 않고 "
            f"최대 {combination_search.max_frontier_size}개 후보만 유지했습니다."
        )

        insight_left, insight_right = st.columns(2)
        with insight_left:
            st.markdown("#### 무승부 가능성 상위")
            st.dataframe(draw_frame, width="stretch", hide_index=True)
        with insight_right:
            st.markdown("#### 이변·불확실성 상위")
            st.dataframe(upset_frame, width="stretch", hide_index=True)
        st.caption(
            "비우세 결과 합계는 최우선 선택 이외 두 결과의 모델 확률 합입니다. "
            "실제 이변 발생을 단정하는 지표가 아닙니다."
        )
        st.download_button(
            "14경기 분석 CSV 다운로드",
            data=result_csv_bytes(toto_result),
            file_name="football_ai_lab_toto14_result.csv",
            mime="text/csv",
            key="toto_download",
        )
        if st.session_state.get("toto_snapshot"):
            st.download_button("14경기 예측 스냅샷 다운로드",st.session_state["toto_snapshot"],file_name="toto_snapshot.json",mime="application/json",key="toto_snapshot_download")

with tactics_tab:
    from tactical_experiment import compare_scenario, exact_match, load_profiles
    from models import normalise_match_dataframe, percentages_summing_to_100

    st.subheader('사용자 전술 자료 가상 실험')
    st.warning('이 자료는 출처와 측정 시점이 확인되지 않은 사용자 제공 가정입니다. 실측 xG나 선수 정보가 아니며, 결과는 예측 정확도가 검증된 확률이 아닙니다.')
    st.caption('기존 예측은 유지하며 선택한 한 경기의 가상 시나리오만 비교합니다. 과거 백테스트, 14경기 조합과 실제 데이터에 자동 적용하지 않습니다.')
    source=st.radio('기존 경기 입력 선택',('프로토 입력','승무패 입력'),horizontal=True,key='tactical_match_source')
    selected_frame=proto_frame if source=='프로토 입력' else toto_frame
    uploaded_profiles=st.file_uploader('사용자 전술 프로필 CSV',type='csv',key='tactical_profiles')
    local_profiles=APP_DIR/'data'/'user_tactical_profiles.csv'
    if uploaded_profiles is None and not local_profiles.is_file():
        st.info('전술 프로필 CSV를 올리세요. 이 환경에는 사용자 자료가 저장되어 있지 않습니다.')
    elif selected_frame is None:
        st.info('먼저 선택한 분석 탭에서 경기 입력을 준비하세요.')
    else:
        try:
            supplied=pd.read_csv(uploaded_profiles if uploaded_profiles is not None else local_profiles,encoding='utf-8-sig')
            profiles=load_profiles(supplied)
            normalized=normalise_match_dataframe(selected_frame)
            st.caption(f'사용자 제공 {len(profiles)}개 팀 · {profiles.league.nunique()}개 리그 · 출처 및 기준시각 미확인')
            index=st.selectbox('가상 실험할 경기',list(range(len(normalized))),
                format_func=lambda i: f"{normalized.iloc[i]['match_id']} · {normalized.iloc[i]['home_team']} vs {normalized.iloc[i]['away_team']}",key='tactical_fixture')
            fixture=normalized.iloc[index].to_dict()
            league=st.selectbox('프로필 리그',sorted(profiles.league.unique()),key='tactical_league')
            league_profiles=profiles[profiles.league==league]
            options=['직접 선택']+[f"{r['team_name_ko']} ({r['team_name_en']})" for _,r in league_profiles.iterrows()]
            rows=list(league_profiles.index)
            home_exact=exact_match(profiles,league,str(fixture['home_team']))
            away_exact=exact_match(profiles,league,str(fixture['away_team']))
            home_default=rows.index(home_exact[0])+1 if len(home_exact)==1 else 0
            away_default=rows.index(away_exact[0])+1 if len(away_exact)==1 else 0
            home_label=st.selectbox('홈팀 프로필',options,index=home_default,key='tactical_home')
            away_label=st.selectbox('원정팀 프로필',options,index=away_default,key='tactical_away')
            if not home_exact or not away_exact:
                st.info('이름이 정확히 일치하지 않는 팀은 직접 선택해 주세요. 유사한 이름만으로 자동 연결하지 않습니다.')
            confirmed=st.checkbox('선택한 프로필이 위 경기의 홈팀·원정팀에 각각 해당함을 확인했습니다.',key='tactical_confirm')
            signature=hashlib.sha256(repr((fixture,home_label,away_label,league,confirmed,config,
                                          profiles.to_json(orient='split',force_ascii=False))).encode()).hexdigest()
            if st.button('가상 시나리오 비교',key='tactical_run'):
                if home_label=='직접 선택' or away_label=='직접 선택':
                    st.error('홈팀과 원정팀 프로필을 선택하세요.')
                else:
                    home=profiles.loc[rows[options.index(home_label)-1]].to_dict()
                    away=profiles.loc[rows[options.index(away_label)-1]].to_dict()
                    result=compare_scenario(fixture,home,away,confirmed=confirmed,config=config)
                    st.session_state['tactical_result']={'signature':signature,'result':result}
            saved=st.session_state.get('tactical_result')
            if isinstance(saved,dict) and saved.get('signature')==signature:
                result=saved['result']
                b=percentages_summing_to_100(result.baseline); e=percentages_summing_to_100(result.experimental)
                table=pd.DataFrame([{'구분':'기존 모델','홈승%':b[0],'무%':b[1],'원정승%':b[2],'예상 스코어':result.baseline_score},
                                    {'구분':'가상 전술 시나리오','홈승%':e[0],'무%':e[1],'원정승%':e[2],'예상 스코어':result.experimental_score}])
                st.dataframe(table,hide_index=True,width='stretch')
                st.write(f'가정 배율: 홈 {result.home_multiplier:.3f}, 원정 {result.away_multiplier:.3f} · rho 원안 {result.raw_rho:.3f}, 적용 {result.safe_rho:.3f}')
                if result.rho_capped: st.info('음수 확률 방지 및 기존 모델 허용 범위 때문에 rho를 제한했습니다.')
                st.caption('적용한 가정: '+', '.join(result.notes)+' · 검증된 실제 확률이 아닙니다.')
        except (ValueError,OSError,pd.errors.ParserError) as exc:
            st.error(f'가상 실험 입력 오류: {exc}')

with year_tab:
    from backtest_2026 import HISTORY_COLUMNS, ROUND_COLUMNS, empty_round_status, run as run_2026_backtest

    st.subheader("2026 승무패 백테스트")
    st.caption("2026-10-27 조사 범위는 1~63회차입니다. 현재 확보한 이력으로 과거 시점을 재구성하며 실제 당시 기록한 예측과 다릅니다.")
    st.info("베트맨 자료를 자동 수집하지 않습니다. 사용·재배포 권한과 경기 순서를 확인한 CSV와 결과 확정시각이 있는 경기 이력을 사용합니다.")
    st.download_button("회차 입력 양식 다운로드", pd.DataFrame(columns=ROUND_COLUMNS).to_csv(index=False).encode("utf-8-sig"),
                       file_name="rounds_2026_template.csv", mime="text/csv", key="year_round_template")
    st.download_button("경기 이력 양식 다운로드", pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(index=False).encode("utf-8-sig"),
                       file_name="history_2026_template.csv", mime="text/csv", key="year_history_template")
    round_upload = st.file_uploader("2026 회차 CSV", type="csv", key="year_round_upload")
    history_upload = st.file_uploader("경기 이력 CSV", type="csv", key="year_history_upload")
    signature = None
    if round_upload is not None and history_upload is not None:
        signature = hashlib.sha256(round_upload.getvalue() + history_upload.getvalue()).hexdigest()
        if st.button("전 회차 예측 및 검증", type="primary", key="year_run"):
            try:
                rounds_input = pd.read_csv(round_upload, encoding="utf-8-sig")
                history_input = pd.read_csv(history_upload, encoding="utf-8-sig")
                with st.spinner("회차별로 예측을 고정하고 검증 중입니다..."):
                    with tempfile.TemporaryDirectory(prefix="football-ai-backtest-") as temp_dir:
                        tables = run_2026_backtest(rounds_input, history_input, Path(temp_dir))
                st.session_state["year_results"] = {"signature": signature, "tables": tables}
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                st.error(f"백테스트 데이터 오류: {exc}")
    else:
        st.caption("두 파일이 모두 있어야 전력 계산을 시작합니다. 빈 양식에는 실제 경기나 점수가 없습니다.")
    stored_year = st.session_state.get("year_results")
    tables = stored_year["tables"] if isinstance(stored_year, dict) and stored_year.get("signature") == signature and signature else None
    show_actual = st.toggle("실제 결과와 성능 공개", value=False, key="year_reveal") if tables else False
    status_table = tables["rounds"].copy() if tables else empty_round_status()
    if not show_actual:
        status_table = status_table.drop(columns=["단식 적중", "TOP20 최고 적중", "TOP20 정답 포함"])
    st.dataframe(status_table, hide_index=True, width="stretch")
    if tables:
        selected = st.selectbox("회차 선택", list(range(1, 64)), key="year_selected_round")
        matches = tables["matches"]
        selected_matches = matches.loc[matches["회차"] == selected]
        if selected_matches.empty:
            st.warning("이 회차는 14경기나 경기 전 전력 자료가 부족하여 분석되지 않았습니다.")
        else:
            prediction_columns = ["번호", "홈팀", "원정팀", "홈승%", "무%", "원정승%", "AI 선택", "두번째", "예상 스코어", "전력 자료"]
            st.dataframe(selected_matches.loc[:, prediction_columns + (["실제 스코어", "실제 결과", "적중"] if show_actual else [])], hide_index=True, width="stretch")
            top = tables["top20"].loc[tables["top20"]["회차"] == selected]
            st.dataframe(top if show_actual else top[["순위", "조합", "확률"]], hide_index=True, width="stretch")
        if show_actual and not matches.empty:
            st.write(f"분석 {len(matches)}경기 · 단식 적중률 {matches['적중'].mean()*100:.2f}% · Brier {matches['Brier Score'].mean():.4f} · Log Loss {matches['Log Loss'].mean():.4f}")
            st.write(f"홈승 기준선 {matches['홈승 기준선 적중'].mean()*100:.2f}% · 과거 빈도 기준선 {matches['빈도 기준선 적중'].mean()*100:.2f}%")
            trend = matches.groupby("회차")["적중"].mean().mul(100)
            st.line_chart(trend)
            st.download_button("경기별 비교 CSV 다운로드", result_csv_bytes(matches), file_name="backtest_2026_matches.csv", mime="text/csv", key="year_download")
    st.caption("검증된 회차만 집계합니다. 당시 이력의 실제 제공 가능 시점, 정정 여부, 투표율과 수익률은 자동 검증하지 않습니다.")

with review_tab:
    st.subheader("승무패 과거검증")
    st.caption("2026년 1~30회차 · 과거 시점 사후 재현. 실제 당시 저장된 예측이 아닙니다.")
    st.warning("확인 가능한 이력만 사용합니다. 오래된 시즌이나 적은 경기 수만 확보된 팀은 신뢰도가 낮습니다. 결과는 수익을 보장하지 않습니다.")
    report_dir = APP_DIR / "reports"
    report_paths = {name: report_dir / f"backtest_2026_1_30_{name}.csv" for name in ("rounds", "matches", "top20")}
    if all(path.is_file() for path in report_paths.values()):
        try:
            report_tables = {name: pd.read_csv(path, encoding="utf-8-sig") for name, path in report_paths.items()}
            report_rounds, report_matches, report_top = (report_tables[name] for name in ("rounds", "matches", "top20"))
            reveal_review = st.toggle("실제 결과와 성적 공개", key="review_reveal", value=False)
            status_columns = ["회차", "확보 경기", "백테스트 상태", "사유"]
            if reveal_review:
                status_columns += ["AI 단식 적중", "홈승 기준선", "TOP20 최고 적중", "TOP20 정답 포함"]
            st.dataframe(report_rounds[status_columns].fillna(""), hide_index=True, width="stretch")
            review_round = st.selectbox("과거 회차 선택", range(1, 31), key="review_round")
            selected_review = report_matches[report_matches["회차"] == review_round]
            selected_top = report_top[report_top["회차"] == review_round]
            if selected_review.empty:
                st.info("공식 대상 경기 또는 경기 전 이력이 아직 검증되지 않은 회차입니다.")
            else:
                prediction_columns = ["번호", "홈팀", "원정팀", "홈승%", "무%", "원정승%", "AI 선택", "예상 스코어", "전력 자료"]
                st.dataframe(selected_review[prediction_columns + (["실제 스코어", "실제 결과", "적중"] if reveal_review else [])], hide_index=True, width="stretch")
                if not selected_top.empty:
                    st.markdown("#### 확률 상위 20개 조합")
                    top_columns = ["순위", "조합", "확률"] + (["일치 경기수", "일치 등급"] if reveal_review else [])
                    st.dataframe(selected_top[top_columns], hide_index=True, width="stretch")
            if reveal_review and not report_matches.empty:
                st.write(f"완료 {report_matches['회차'].nunique()}회차 · {len(report_matches)}경기 · AI 적중률 {report_matches['적중'].mean()*100:.2f}% · 홈승 기준선 {report_matches['홈승 기준선 적중'].mean()*100:.2f}%")
                st.write(f"Brier {report_matches['Brier Score'].mean():.4f} · Log Loss {report_matches['Log Loss'].mean():.4f}")
                st.line_chart(report_matches.groupby("회차")["적중"].mean().mul(100))
                league = report_matches.groupby("리그")["적중"].agg(["size", "mean"]).rename(columns={"size":"경기 수", "mean":"적중률"})
                st.markdown("#### 리그별 성적")
                st.dataframe(league, width="stretch")
                monthly = report_matches.assign(월=report_matches["날짜"].str[:7]).groupby("월")["적중"].agg(["size", "mean"]).rename(columns={"size":"경기 수", "mean":"적중률"})
                st.markdown("#### 월별 성적")
                st.dataframe(monthly, width="stretch")
                st.download_button("전체 경기 결과 CSV 다운로드", result_csv_bytes(report_matches), file_name="backtest_2026_1_30_matches.csv", mime="text/csv", key="review_download")
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"보고서를 읽을 수 없습니다: {exc}")
    else:
        st.info("이 환경에 검증된 1~30회차 보고서가 없습니다. 로컬 보고서를 생성하면 이 화면에 표시됩니다.")

    st.markdown('### 4대 리그 무승부 연구')
    research_dir=APP_DIR/'reports'/'local_backtest_2026'/'4대리그_무승부_2026'
    stats_path=research_dir/'리그별_무승부_결과.csv'
    model_path=research_dir/'전략별_성능.csv'
    if stats_path.is_file() and model_path.is_file():
        st.caption('경기 결과 기술 통계는 2026-07-28부터 조회시각까지입니다. 예측 비교는 확인 가능한 오래된 일부 이력만 사용한 사후 연구이며, 맹검 독립 검증이 아닙니다.')
        st.dataframe(pd.read_csv(stats_path,encoding='utf-8-sig'),hide_index=True,width='stretch')
        st.caption('A 기본 최고 확률 · B 기본 확률과 후보 표시 · D 확률을 바꾸지 않은 선택 규칙. C는 충분한 검증 학습 이력이 없어 미실행입니다.')
        st.dataframe(pd.read_csv(model_path,encoding='utf-8-sig'),hide_index=True,width='stretch')
        score_path=research_dir/'전략별_경기평가.csv'
        if score_path.is_file():
            score_data=pd.read_csv(score_path,encoding='utf-8-sig')
            if not score_data.empty:
                score_data['월']=pd.to_datetime(score_data['날짜'],utc=True).dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m')
                monthly=score_data.groupby(['월','모델'])['적중'].mean().mul(100).unstack()
                st.caption('월별 단식 적중률(%) · 확인된 EPL·세리에 A 일부 표본의 사후 비교')
                st.line_chart(monthly)
        quality_path=research_dir/'분석가능_데이터품질.csv'
        if quality_path.is_file():
            with st.expander('분석 표본의 데이터 충분성'):
                st.dataframe(pd.read_csv(quality_path,encoding='utf-8-sig'),hide_index=True,width='stretch')
                coverage_path=research_dir/'이전시즌_이력확보현황.csv'
                if coverage_path.is_file():
                    st.caption('2026-01-02 이전에 시작한 이전 시즌의 현재 완료 기록 중 공식 대조한 이력 비율입니다. 미대조 경기의 당시 공개 여부는 확인되지 않았습니다.')
                    st.dataframe(pd.read_csv(coverage_path,encoding='utf-8-sig'),hide_index=True,width='stretch')
        detail_path=research_dir/'경기별_무승부_진단.csv'
        if detail_path.is_file():
            detail=pd.read_csv(detail_path,encoding='utf-8-sig')
            if not detail.empty:
                idx=st.selectbox('연구 경기 상세',list(range(len(detail))),key='four_league_match',
                    format_func=lambda i: f"{detail.iloc[i]['league']} · {detail.iloc[i]['home_team']} vs {detail.iloc[i]['away_team']}")
                row=detail.iloc[idx]
                left,mid,right=st.columns(3)
                left.metric('홈승',f"{row['홈승 확률']*100:.1f}%")
                mid.metric('무승부',f"{row['무승부 확률']*100:.1f}%")
                right.metric('원정승',f"{row['원정승 확률']*100:.1f}%")
                st.write(f"무승부 후보: {'예' if row['B 후보'] else '아니오'} · 근거: {row['후보 근거']}")
                kickoff_label=pd.Timestamp(row['kickoff']).tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M')
                st.caption(f"팀 이력 홈 {row['home_history']}경기 / 원정 {row['away_history']}경기 · "
                           f"데이터 부족 {'예' if row['insufficient'] else '아니오'} · 경기 시각 {kickoff_label} KST")
                st.caption('표시된 승·무·패는 기존 모델 확률이며 후보는 별도의 규칙 판단입니다. 사후 분석에는 2026-01-02까지 확인된 일부 과거 이력만 사용했습니다.')
                if st.toggle('연구 경기 실제 결과 공개',key='four_league_reveal'):
                    st.write(f"실제 결과: {row['actual']} · 기본 선택: {row['A 선택']} · 선택 규칙 D: {row['D 선택']}")
    else:
        st.info('이 환경에는 로컬 4대 리그 연구 결과가 없습니다. 경기 원본은 저장소에 포함되지 않습니다.')

    from draw_diagnostics import RESEARCH_NOTICE
    from draw_research_workflow import create_predictions, reveal_results

    st.markdown("### 실험: 무승부 진단 비교")
    st.caption(RESEARCH_NOTICE)
    st.info("이 분석은 베트맨 사전 투표율, 베팅 비율 또는 배당률을 참고하지 않습니다. Football AI Lab의 자체 축구 통계 모델만 사용합니다.")
    st.warning("2026년 1·2회차 재분석은 이미 결과를 확인한 데이터를 대상으로 한 사후 연구 실험입니다. 독립 검증 결과나 실제 당시 예측 성적이 아닙니다.")
    st.caption("신규 진단은 소표본 실험입니다. 학습 기준시각 이전에 실제로 확인 가능한 경기 이력만 사용해야 합니다.")
    local_research = APP_DIR / 'reports' / 'local_backtest_2026' / '무승부_진단_연구' / '1_2회차_경기별_모델비교.csv'
    local_metrics = local_research.parent / '1_2회차_모델별_성능.csv'
    if local_research.is_file() and local_metrics.is_file():
        local_results = pd.read_csv(local_research, encoding='utf-8-sig')
        local_summary = pd.read_csv(local_metrics, encoding='utf-8-sig')
        local_show = st.toggle('1·2회차 실제 결과와 실험 성적 공개', key='draw_local_reveal')
        visible = ['회차','경기번호','홈팀','원정팀','기존 무%','보강 무%','개선 무%',
                   '기존 선택','보강 선택','개선 선택','위험 점수','후보','후보 근거','홈 이력수','원정 이력수']
        if local_show: visible += ['실제','실제 스코어','개선 효과']
        st.dataframe(local_results[visible],hide_index=True,width='stretch')
        if local_show:
            st.caption('A 기존 / B 이력 보강 / C 진단 확률 / D 보강 확률 유지와 후보 표시')
            st.dataframe(local_summary,hide_index=True,width='stretch')
            st.download_button('1·2회차 비교 CSV 다운로드',result_csv_bytes(local_results),
                               file_name='draw_research_1_2.csv',mime='text/csv',key='draw_local_download')
    else:
        st.caption('현재 환경에는 로컬 1·2회차 실험 결과가 없습니다. 아래의 검증된 사용자 자료로 분석할 수 있습니다.')
    with st.expander('검증된 CSV로 실험 실행'):
        st.caption('회차 CSV와 기존·보강 이력 CSV를 각각 올리세요. 지정 양식 외 열은 거부하며, 이력의 확보시각은 당시 확인 근거가 있어야 합니다.')
        draw_rounds=st.file_uploader('공식 검증된 회차 CSV',type='csv',key='draw_rounds')
        draw_old=st.file_uploader('기존 경기 이력 CSV',type='csv',key='draw_old')
        draw_extra=st.file_uploader('보강 경기 이력 CSV',type='csv',key='draw_extra')
        draw_round=st.selectbox('실험 회차', [1,2],key='draw_round')
        if draw_rounds and draw_old and draw_extra:
            draw_signature=hashlib.sha256(draw_rounds.getvalue()+draw_old.getvalue()+draw_extra.getvalue()+str(draw_round).encode()).hexdigest()
            if st.button('경기 전 예측 고정',key='draw_predict'):
                try:
                    rr=pd.read_csv(draw_rounds,encoding='utf-8-sig')
                    old=pd.read_csv(draw_old,encoding='utf-8-sig')
                    extra=pd.read_csv(draw_extra,encoding='utf-8-sig')
                    with st.spinner('경기 전 이력으로 무승부 신호를 계산하고 있습니다...'):
                        payload,answers=create_predictions(rr,old,extra,draw_round)
                    st.session_state['draw_experiment']={'signature':draw_signature,'prediction':payload,'answers':answers}
                except (ValueError, OSError, pd.errors.ParserError) as exc:
                    st.error(f'무승부 진단 자료 오류: {exc}')
            saved=st.session_state.get('draw_experiment')
            if saved and saved['signature']==draw_signature:
                p=saved['prediction']
                st.caption(f"학습 {p['training_count']}경기 · 무승부 {p['training_draws']}경기 · 종료 {p['training_end']} · 후보 기준은 학습 점수 상위 25%")
                st.dataframe(pd.DataFrame([{k:v for k,v in row.items() if k not in ('홈 최근5','원정 최근5','홈 최근10','원정 최근10')}
                                           for row in p['predictions']]),hide_index=True,width='stretch')
                if st.toggle('고정된 예측의 실제 결과 공개',key='draw_reveal'):
                    try:
                        detail,summary=reveal_results(p,saved['answers'])
                        st.dataframe(summary,hide_index=True,width='stretch')
                        st.dataframe(detail,hide_index=True,width='stretch')
                        st.download_button('무승부 진단 결과 CSV 다운로드',result_csv_bytes(detail),
                                           file_name='draw_diagnostics.csv',mime='text/csv',key='draw_download')
                    except ValueError as exc:
                        st.error(f'예측 검증 오류: {exc}')

with st.expander("CSV 형식 안내"):
    st.markdown(
        """
        필수 열은 `홈팀`, `원정팀`, `홈 기대득점`, `원정 기대득점`입니다.
        선택 열은 `경기번호`, `홈 Elo`, `원정 Elo`, `홈승 배당`, `무승부 배당`,
        `원정승 배당`, `데이터 구분`입니다. 영문 헤더도 지원합니다. 배당이 없으면
        기대수익률은 빈 값으로 표시됩니다.
        """
    )

with st.expander("실제 데이터 API 안내"):
    st.markdown(
        """
        football-data.org v4 무료 플랜의 실제 일정·최종 결과·팀 식별 정보를 사용합니다.
        무료 플랜은 분당 요청 제한이 있으므로 동일 리그·시즌 응답을 15분간 캐시합니다.
        API 오류나 제한 초과 시에는 **CSV 업로드**로 계속 분석할 수 있습니다.

        API가 제공한 최종 스코어와 앱이 계산한 기대득점은 서로 다른 데이터입니다.
        앱의 기대득점은 예측 경기 시작 전 완료 결과만 사용한 모델 추정치이며 실측 xG가
        아닙니다. 무료 응답에 배당이 없으면 기대수익률은 계산하지 않습니다.
        """
    )

st.caption("Football data provided by the Football-Data.org API")

with st.expander("갤럭시 홈 화면에 추가"):
    st.markdown(
        """
        1. 삼성 인터넷 또는 Chrome에서 배포된 앱 주소를 엽니다.
        2. 브라우저 메뉴에서 **홈 화면에 추가**를 선택합니다.
        3. 이름을 확인하고 **추가**를 누릅니다.

        Streamlit 단독 앱은 서비스 워커·오프라인 캐시·설치 매니페스트를 완전히
        제어하는 PWA가 아닙니다. 이 MVP는 홈 화면 바로가기 방식이며, 실행 시
        인터넷 연결과 Streamlit 서버가 필요합니다.
        """
    )
