"""Football AI Lab Streamlit 애플리케이션."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

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


def uploaded_or_demo_data(prefix: str) -> tuple[pd.DataFrame | None, str]:
    source = st.radio(
        "데이터 선택",
        ("14경기 가상 데모", "CSV 업로드"),
        horizontal=True,
        key=f"{prefix}_source",
    )
    if source == "14경기 가상 데모":
        st.markdown('<span class="data-label">가상 데이터</span>', unsafe_allow_html=True)
        st.caption("팀·기대득점·Elo·배당은 기능 시험을 위해 만든 값이며 실제 경기가 아닙니다.")
        return load_demo_data(), "가상 데이터"

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

proto_tab, toto_tab = st.tabs(("프로토 분석", "승무패 14경기"))

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
        st.dataframe(
            proto_result.loc[
                :,
                [
                    "경기번호",
                    "홈승 적정배당",
                    "무승부 적정배당",
                    "원정승 적정배당",
                    "홈승 기대수익률(%)",
                    "무승부 기대수익률(%)",
                    "원정승 기대수익률(%)",
                    "데이터 구분",
                ],
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption("기대수익률 = 모델 확률 × 입력 배당 − 1. 양수라도 실제 수익을 보장하지 않습니다.")
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

with st.expander("CSV 형식 안내"):
    st.markdown(
        """
        필수 열은 `홈팀`, `원정팀`, `홈 기대득점`, `원정 기대득점`입니다.
        선택 열은 `경기번호`, `홈 Elo`, `원정 Elo`, `홈승 배당`, `무승부 배당`,
        `원정승 배당`, `데이터 구분`입니다. 영문 헤더도 지원합니다. 배당이 없으면
        기대수익률은 빈 값으로 표시됩니다.
        """
    )

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
