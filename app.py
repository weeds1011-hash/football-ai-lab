"""Football AI Lab Streamlit 애플리케이션."""

from __future__ import annotations

from datetime import datetime
import hashlib
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
) -> tuple[dict[str, object], dict[str, int | None]]:
    """같은 리그·시즌 API 응답을 15분간 캐시한다."""

    response = FootballDataClient(_api_key).competition_matches(
        competition_code, season
    )
    return dict(response.data), {
        "remaining": response.rate_limit.remaining_per_minute,
        "reset_seconds": response.rate_limit.reset_seconds,
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

proto_tab, toto_tab, year_tab = st.tabs(("프로토 분석", "승무패 14경기", "2026 승무패 백테스트"))

with proto_tab:
    st.subheader("프로토 경기 분석")
    proto_frame, proto_data_type = uploaded_or_demo_data("proto")
    if proto_frame is not None:
        show_input_preview(proto_frame)
        if st.button("프로토 분석 실행", type="primary", key="proto_run"):
            try:
                _, proto_result, _ = analyze_dataframe(
                    proto_frame, config, default_data_type=proto_data_type
                )
                st.session_state["proto_result"] = proto_result
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

with toto_tab:
    st.subheader("축구토토 승무패 14경기")
    toto_frame, toto_data_type = uploaded_or_demo_data("toto")
    if toto_frame is not None:
        show_input_preview(toto_frame)
        if st.button("14경기 조합 분석", type="primary", key="toto_run"):
            try:
                _, toto_result, toto_analyses = analyze_dataframe(
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
