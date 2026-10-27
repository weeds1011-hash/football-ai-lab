"""Football AI Lab Streamlit 애플리케이션."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from backtesting import (
    PRODUCT_PROTO, PRODUCT_TOTO, aggregate_evaluations, backtest_template_csv,
    create_prediction_snapshot, evaluate_snapshot, history_template_csv,
    prediction_display_frame, prepare_round, product_rows, read_backtest_file,
    read_history_file, reconstructed_model_input, round_status_frame,
    snapshot_model_input,
)
from betman_data import (
    BETMAN_HOME_URL,
    BETMAN_TERMS_URL,
    MARKET_1X2,
    available_rounds,
    betman_template_csv,
    display_betman_frame,
    latest_retrieved_at,
    link_betman_matches,
    read_betman_file,
    read_team_alias_file,
    select_round,
    team_alias_template_csv,
    validate_toto_round,
)
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
DEMO_BACKTEST_CSV = APP_DIR / "data" / "demo_backtest_rounds.csv"
FREE_COMPETITION_BY_CODE = {item.code: item for item in FREE_COMPETITIONS}

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


@st.cache_data(show_spinner=False)
def load_betman_upload(payload: bytes, filename: str) -> pd.DataFrame:
    """사용자가 제공한 베트맨 CSV/XLSX를 파일 내용 기준으로 캐시한다."""

    return read_betman_file(payload, filename)


@st.cache_data(show_spinner=False)
def load_team_alias_upload(payload: bytes, filename: str) -> pd.DataFrame:
    return read_team_alias_file(payload, filename)


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


def link_betman_to_model(
    selected_frame: pd.DataFrame,
    alias_frame: pd.DataFrame | None,
    api_key: str,
):
    """선택 경기 날짜·대회만 조회해 베트맨 행을 모델 입력으로 연결한다."""

    fixture_frames: list[pd.DataFrame] = []
    supported_codes = set(FREE_COMPETITION_BY_CODE)
    groups = selected_frame.loc[:, ["competition_code", "kickoff_kst"]].copy()
    groups["competition_code"] = groups["competition_code"].astype(str).str.upper()
    groups["target_date"] = groups["kickoff_kst"].map(lambda value: value.date())
    groups = groups.loc[groups["competition_code"].isin(supported_codes)].drop_duplicates(
        ["competition_code", "target_date"]
    )

    for _, group in groups.iterrows():
        code = group["competition_code"]
        target_date = group["target_date"]
        season = season_start_year(target_date, code)
        payload, _ = load_real_api_payload(code, season, _api_key=api_key)
        bundle = build_real_match_bundle(
            payload,
            target_date=target_date,
            now=datetime.now(SEOUL_TIMEZONE),
        )
        if not bundle.input_frame.empty:
            fixture_frames.append(bundle.input_frame)

    fixtures = (
        pd.concat(fixture_frames, ignore_index=True)
        if fixture_frames
        else pd.DataFrame()
    )
    return link_betman_matches(
        selected_frame,
        fixtures,
        alias_frame=alias_frame,
        supported_codes=supported_codes,
    )


def betman_user_file_data(prefix: str) -> tuple[pd.DataFrame | None, str]:
    """베트맨 자동 수집 대신 사용자 제공 회차 파일을 안전하게 처리한다."""

    is_toto = prefix == "toto"
    st.markdown(
        '<span class="real-label">베트맨 실제 자료 · 사용자 파일</span>',
        unsafe_allow_html=True,
    )
    st.warning(
        "베트맨 공식 API와 자동 수집 허가를 확인하지 못해 사이트를 자동 스크래핑하지 않습니다. "
        "이용 권한을 확인해 직접 확보한 CSV/XLSX만 업로드하세요."
    )
    st.caption(
        "출처: betman.co.kr · 파일의 회차, 경기번호, 일정, 배당·투표율과 조회 기준시각은 "
        "업로드한 자료를 그대로 사용하며 임의 값을 만들지 않습니다."
    )
    template_left, template_right = st.columns(2)
    with template_left:
        st.download_button(
            "베트맨 입력 템플릿",
            data=betman_template_csv(),
            file_name="betman_round_template.csv",
            mime="text/csv",
            key=f"{prefix}_betman_template",
        )
    with template_right:
        st.download_button(
            "팀명 매핑 템플릿",
            data=team_alias_template_csv(),
            file_name="betman_team_alias_template.csv",
            mime="text/csv",
            key=f"{prefix}_betman_alias_template",
        )

    uploaded = st.file_uploader(
        "베트맨 회차 CSV 또는 XLSX",
        type=("csv", "xlsx"),
        help="베트맨 자료의 사용·재사용 권한은 업로드한 사용자가 확인해야 합니다.",
        key=f"{prefix}_betman_upload",
    )
    if uploaded is None:
        st.info("회차 파일을 업로드하면 회차 선택과 팀 매칭을 진행할 수 있습니다.")
        return None, "베트맨 사용자 제공 자료"

    payload = uploaded.getvalue()
    try:
        frame = load_betman_upload(payload, uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        return None, "베트맨 사용자 제공 자료"

    product_text = frame["product"].str.lower()
    if is_toto and product_text.str.contains("승무패", regex=False).any():
        frame = frame.loc[product_text.str.contains("승무패", regex=False)].copy()
    elif not is_toto and product_text.str.contains("프로토", regex=False).any():
        frame = frame.loc[product_text.str.contains("프로토", regex=False)].copy()
    if frame.empty:
        st.error("선택한 분석 탭에 해당하는 상품 행이 없습니다.")
        return None, "베트맨 사용자 제공 자료"

    rounds = available_rounds(frame)
    selected_round = st.selectbox("회차", rounds, key=f"{prefix}_betman_round")
    round_frame = select_round(frame, selected_round)

    if not is_toto:
        sports = round_frame["sport"].drop_duplicates().tolist()
        default_sport_index = next(
            (index for index, value in enumerate(sports) if "축구" in value), 0
        )
        selected_sport = st.selectbox(
            "종목",
            sports,
            index=default_sport_index,
            key=f"{prefix}_betman_sport",
        )
        round_frame = round_frame.loc[round_frame["sport"] == selected_sport].copy()
        dates = sorted({value.date().isoformat() for value in round_frame["kickoff_kst"]})
        selected_date = st.selectbox(
            "경기 날짜",
            ["전체", *dates],
            key=f"{prefix}_betman_date",
        )
        if selected_date != "전체":
            round_frame = round_frame.loc[
                round_frame["kickoff_kst"].map(lambda value: value.date().isoformat())
                == selected_date
            ].copy()

    st.caption(f"파일 갱신 기준시각: {latest_retrieved_at(round_frame)}")
    st.dataframe(display_betman_frame(round_frame), width="stretch", hide_index=True)

    ineligible = round_frame.loc[round_frame["market_type"] != MARKET_1X2]
    if not ineligible.empty:
        st.info(
            f"핸디캡·언더오버·기타 시장 {len(ineligible)}개는 별도 모델이 없어 "
            "목록만 표시하며 1X2 분석에 연결하지 않습니다."
        )

    eligible = round_frame.loc[round_frame["market_type"] == MARKET_1X2].copy()
    if is_toto:
        try:
            validate_toto_round(eligible)
        except ValueError as exc:
            st.error(str(exc))
            st.dataframe(
                pd.DataFrame(
                    {
                        "확보 경기번호": eligible["match_no"].tolist(),
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            return None, "베트맨 사용자 제공 자료"
        selected_frame = eligible
    else:
        options = eligible.index.tolist()
        chosen = st.multiselect(
            "AI 분석할 일반 승무패 경기",
            options,
            default=options,
            format_func=lambda index: (
                f"{int(eligible.loc[index, 'match_no'])}번 · "
                f"{eligible.loc[index, 'home_team']} vs {eligible.loc[index, 'away_team']}"
            ),
            key=f"{prefix}_betman_matches",
        )
        selected_frame = eligible.loc[chosen].copy() if chosen else eligible.iloc[0:0].copy()
        if selected_frame.empty:
            st.info("일반 승무패 경기를 한 개 이상 선택하세요.")
            return None, "베트맨 사용자 제공 자료"

    alias_upload = st.file_uploader(
        "확인된 팀명 매핑 CSV/XLSX (선택)",
        type=("csv", "xlsx"),
        help="유사도 자동 매칭은 하지 않습니다. 베트맨 팀명과 football-data 팀명을 직접 확인해 입력하세요.",
        key=f"{prefix}_betman_alias_upload",
    )
    alias_frame = None
    alias_payload = b""
    if alias_upload is not None:
        alias_payload = alias_upload.getvalue()
        try:
            alias_frame = load_team_alias_upload(alias_payload, alias_upload.name)
        except ValueError as exc:
            st.error(str(exc))
            return None, "베트맨 사용자 제공 자료"

    signature_source = (
        payload
        + alias_payload
        + selected_round.encode("utf-8")
        + ",".join(str(value) for value in selected_frame.index).encode("ascii")
    )
    signature = hashlib.sha256(signature_source).hexdigest()
    state_key = f"{prefix}_betman_link"
    api_key = football_data_api_key()
    button_label = (
        "football-data.org 매칭 및 14경기 준비"
        if is_toto
        else "football-data.org 매칭 및 선택 경기 준비"
    )
    if st.button(button_label, type="primary", key=f"{prefix}_betman_link_run"):
        if not api_key:
            st.error("football-data.org API 키가 없어 전력 매칭을 실행할 수 없습니다.")
        else:
            try:
                with st.spinner("공식 일정과 이전 결과를 조회하고 팀을 확인하는 중입니다..."):
                    link_result = link_betman_to_model(selected_frame, alias_frame, api_key)
                st.session_state[state_key] = {
                    "signature": signature,
                    "selected_count": len(selected_frame),
                    "result": link_result,
                }
            except FootballDataRateLimitError as exc:
                st.error(f"football-data.org 요청 한도를 초과했습니다: {exc}")
            except (FootballDataError, ValueError) as exc:
                st.error(f"경기 매칭 오류: {exc}")

    stored = st.session_state.get(state_key)
    if not isinstance(stored, dict) or stored.get("signature") != signature:
        st.info("회차와 경기를 확인한 뒤 football-data.org 매칭 버튼을 누르세요.")
        return None, "베트맨 사용자 제공 자료"

    link_result = stored["result"]
    st.markdown("#### 팀·일정 매칭 결과")
    st.dataframe(link_result.status_frame, width="stretch", hide_index=True)
    matched = link_result.matched_count
    selected_count = int(stored["selected_count"])
    if is_toto and matched != 14:
        st.error(
            f"14경기 중 {matched}경기만 안전하게 매칭됐습니다. "
            "매칭 실패·미지원 경기를 해결하기 전에는 조합 분석을 실행하지 않습니다."
        )
        return None, "베트맨 사용자 제공 자료"
    if matched == 0:
        st.error("안전하게 매칭된 경기가 없습니다. 대회코드와 팀명 매핑을 확인하세요.")
        return None, "베트맨 사용자 제공 자료"
    if matched < selected_count:
        st.warning(
            f"선택한 {selected_count}경기 중 {matched}경기만 분석에 연결합니다. "
            "나머지는 임의 추정하지 않았습니다."
        )
    else:
        st.success(f"{matched}경기를 football-data.org 전력 자료와 안전하게 연결했습니다.")
    return link_result.model_frame, "베트맨 사용자 파일 + football-data.org 모델 추정"


def uploaded_or_demo_data(prefix: str) -> tuple[pd.DataFrame | None, str]:
    betman_option = "베트맨 승무패 회차" if prefix == "toto" else "베트맨 실제 경기"
    source = st.radio(
        "데이터 선택",
        ("14경기 가상 데모", "실제 데이터 API", betman_option, "CSV 업로드"),
        horizontal=True,
        key=f"{prefix}_source",
    )
    active_source_key = f"{prefix}_active_source"
    if st.session_state.get(active_source_key) != source:
        st.session_state[active_source_key] = source
        st.session_state.pop("proto_result" if prefix == "proto" else "toto_bundle", None)
        st.session_state.pop(f"{prefix}_betman_link", None)
    if source == "14경기 가상 데모":
        st.markdown('<span class="data-label">가상 데이터</span>', unsafe_allow_html=True)
        st.caption("팀·기대득점·Elo·배당은 기능 시험을 위해 만든 값이며 실제 경기가 아닙니다.")
        return load_demo_data(), "가상 데이터"
    if source == "실제 데이터 API":
        return real_api_data(prefix)
    if source == betman_option:
        return betman_user_file_data(prefix)

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


def snapshot_download_bytes(snapshot) -> bytes:
    """정답을 포함하지 않는 예측과 재현 메타데이터를 내보낸다."""
    content = {
        "snapshot_id": snapshot.snapshot_id,
        "product": snapshot.product,
        "round_id": snapshot.round_id,
        "cutoff": snapshot.cutoff.isoformat(),
        "cutoff_basis": snapshot.cutoff_basis,
        "model_version": snapshot.model_version,
        "model_config": snapshot.model_config,
        "mode": snapshot.mode,
        "created_at": snapshot.created_at.isoformat(),
        "source": snapshot.source_label,
        "acquired_at": snapshot.acquired_at_label,
        "history_match_count": snapshot.history_match_count,
        "history_start": snapshot.history_start,
        "history_end": snapshot.history_end,
        "history_sources": snapshot.history_sources,
        "predictions": snapshot.predictions.to_dict(orient="records"),
        "top_combinations": (
            [{"outcomes": list(labels), "probability": probability}
             for labels, probability in snapshot.top_combinations.combinations]
            if snapshot.top_combinations else []
        ),
    }
    return json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")


def show_backtest_metrics(evaluation) -> None:
    st.write(
        f"분석 {evaluation.total_matches}경기 · 적중 {evaluation.correct_matches}경기 · "
        f"적중률 {evaluation.accuracy * 100:.2f}%"
    )
    st.write(f"다중분류 Brier Score {evaluation.brier_score:.4f} · Log Loss {evaluation.log_loss:.4f}")
    st.caption("Brier Score는 세 결과의 제곱오차 합의 경기 평균입니다. 두 점수는 낮을수록 좋습니다.")


def show_backtest_tab(config: ModelConfig) -> None:
    st.subheader("과거 회차 백테스트")
    st.caption("현재 확보한 과거 자료로 경기 전 시점을 재구성합니다. 적중률, 확률 보정, 수익성은 서로 다릅니다.")
    source = st.radio("백테스트 자료", ("가상 테스트 자료", "사용자 과거 회차 파일"), key="backtest_source")
    if source == "가상 테스트 자료":
        st.warning("가상 데이터: 경기·점수·전력 수치는 기능 시험용이며 실제 회차가 아닙니다.")
        payload = DEMO_BACKTEST_CSV.read_bytes()
        filename = DEMO_BACKTEST_CSV.name
    else:
        st.info("베트맨 자동 수집은 제공하지 않습니다. 이용 권한을 확인한 CSV/XLSX를 업로드하세요.")
        st.download_button("과거 회차 빈 템플릿", backtest_template_csv(),
                           file_name="backtest_round_template.csv", mime="text/csv", key="bt_template")
        upload = st.file_uploader("과거 회차 CSV 또는 XLSX", type=("csv", "xlsx"), key="bt_upload")
        if upload is None:
            st.info("완료된 경기와 결과를 포함한 파일을 선택하세요.")
            return
        payload, filename = upload.getvalue(), upload.name
    try:
        frame = read_backtest_file(payload, filename)
    except (ValueError, OSError) as exc:
        st.error(f"과거 회차 파일 오류: {exc}")
        return
    product = st.radio("상품", (PRODUCT_PROTO, PRODUCT_TOTO), key="bt_product", horizontal=True)
    status = round_status_frame(frame, product)
    if status.empty:
        st.info("이 상품에 해당하는 회차가 없습니다.")
        return
    st.dataframe(status, width="stretch", hide_index=True)
    round_ids = status["회차"].tolist()
    selected_round = st.selectbox("과거 회차", round_ids, key="bt_round")
    try:
        prepared = prepare_round(frame, product, selected_round)
    except ValueError as exc:
        st.error(f"회차 불완전: {exc}")
        return
    context = prepared.context
    st.caption(f"기준시각: {context.cutoff.strftime('%Y-%m-%d %H:%M %Z')} · {context.cutoff_basis}")
    st.caption(f"회차 자료 출처: {context.source_label} · 확보 기준시각: {context.acquired_at_label}")
    st.dataframe(context.target_frame.loc[:, ["match_no", "kickoff_kst", "league", "home_team", "away_team"]].rename(
        columns={"match_no": "경기번호", "kickoff_kst": "경기시각", "league": "리그",
                 "home_team": "홈팀", "away_team": "원정팀"}), width="stretch", hide_index=True)
    mode = st.radio("전력 자료", ("경기 전 입력값으로 현재 재계산", "결과 확정시각이 있는 과거 이력으로 재구성"), key="bt_mode")
    history_payload = b""
    history_filename = ""
    if mode.startswith("경기 전 입력값"):
        st.caption("파일의 스냅샷시각과 전력 입력을 검증해 지금의 모델로 다시 계산합니다. 당시 저장된 예측의 증거는 아닙니다. 원래 저장된 예측 가져오기는 아직 지원하지 않습니다.")
    else:
        st.caption("경기 시작·결과 확정·출처·확보시각을 포함한 사용자 이력이 필요합니다. 기준시각 이후 확정된 결과는 제외합니다.")
        st.download_button("과거 이력 빈 템플릿", history_template_csv(), file_name="backtest_history_template.csv",
                           mime="text/csv", key="bt_history_template")
        history_upload = st.file_uploader("경기 이력 CSV 또는 XLSX", type=("csv", "xlsx"), key="bt_history")
        if history_upload is not None:
            history_payload, history_filename = history_upload.getvalue(), history_upload.name
    signature = hashlib.sha256(payload + history_payload + product.encode() + selected_round.encode()
                               + mode.encode() + repr(config).encode()).hexdigest()
    if st.button("경기 전 시점 AI 분석", type="primary", key="bt_predict"):
        try:
            with st.spinner("기준시각 이전 자료로 예측을 고정하고 있습니다..."):
                if mode.startswith("경기 전 입력값"):
                    model_input = snapshot_model_input(context)
                else:
                    if not history_payload:
                        raise ValueError("과거 이력 파일을 선택하세요.")
                    model_input = reconstructed_model_input(context, read_history_file(history_payload, history_filename))
                snapshot = create_prediction_snapshot(context, model_input, config)
            st.session_state["bt_frozen"] = {"signature": signature, "snapshot": snapshot,
                                             "outcomes": prepared.outcomes, "evidence": model_input.evidence_frame}
            st.session_state.pop("bt_evaluation", None)
        except (ValueError, FootballDataError) as exc:
            st.error(f"백테스트 준비 오류: {exc}")
    frozen = st.session_state.get("bt_frozen")
    if isinstance(frozen, dict) and frozen.get("signature") == signature:
        snapshot = frozen["snapshot"]
        st.success(f"예측 고정 완료 · ID {snapshot.snapshot_id} · 세션에 저장됨")
        st.caption(f"모델 {snapshot.model_version} · {snapshot.mode} · 사용 이력 {snapshot.history_match_count}경기 "
                   f"({snapshot.history_start} ~ {snapshot.history_end})")
        st.caption(f"경기 이력 출처·확보시각: {snapshot.history_sources}")
        st.dataframe(prediction_display_frame(snapshot), width="stretch", hide_index=True)
        with st.expander("전력 입력 근거"):
            st.dataframe(frozen["evidence"], width="stretch", hide_index=True)
        if snapshot.top_combinations is not None:
            st.markdown("#### 예측 상위 20개 조합")
            st.dataframe(snapshot.top_combinations.to_frame(), width="stretch", hide_index=True)
        st.download_button("결과 없는 예측 스냅샷 JSON 다운로드", snapshot_download_bytes(snapshot),
                           file_name="backtest_prediction.json", mime="application/json", key="bt_snapshot_download")
        st.caption("세션 저장은 브라우저 새로고침이나 서버 재시작 후 유지되지 않을 수 있습니다. JSON을 보관하세요.")
        if st.button("실제 결과 공개 및 비교", key="bt_reveal"):
            try:
                st.session_state["bt_evaluation"] = {"signature": signature,
                    "result": evaluate_snapshot(snapshot, frozen["outcomes"])}
            except ValueError as exc:
                st.error(f"결과 비교 오류: {exc}")
        evaluation_state = st.session_state.get("bt_evaluation")
        if isinstance(evaluation_state, dict) and evaluation_state.get("signature") == signature:
            evaluation = evaluation_state["result"]
            show_backtest_metrics(evaluation)
            st.dataframe(evaluation.match_frame, width="stretch", hide_index=True)
            st.markdown("#### 리그별 성능")
            st.dataframe(evaluation.league_frame, width="stretch", hide_index=True)
            st.markdown("#### 날짜·회차별 성능")
            st.dataframe(evaluation.trend_frame, width="stretch", hide_index=True)
            if evaluation.actual_combination is not None:
                st.write(f"실제 조합: {evaluation.actual_combination}")
                st.write(f"상위 20개 포함: {'예' if evaluation.actual_in_top20 else '아니요'} · "
                         f"최상위 조합 일치 {evaluation.top1_matching_matches}/14 · "
                         f"전체 적중: {'예' if evaluation.all_14_correct else '아니요'}")
            st.download_button("백테스트 비교 결과 CSV 다운로드", result_csv_bytes(evaluation.match_frame),
                               file_name="backtest_evaluation.csv", mime="text/csv", key="bt_result_download")
    else:
        st.info("회차와 자료를 확인한 뒤 경기 전 시점 AI 분석을 실행하세요.")

    with st.expander("여러 회차 백테스트"):
        st.caption("경기 전 입력 스냅샷이 있는 완료 회차를 대상으로 모든 예측을 먼저 고정하고 결과를 비교합니다.")
        batch_size = st.selectbox("최근 회차 수", (5, 10, 20), key="bt_batch_size")
        completed = status.loc[status["상태"] == "완료", "회차"].tolist()
        completed.sort(key=lambda rid: product_rows(frame, product).loc[
            product_rows(frame, product)["round_id"] == rid, "kickoff_kst"].min(), reverse=True)
        chosen = completed[:batch_size]
        st.caption(f"요청 {batch_size}회차 중 파일에서 완료된 {len(chosen)}회차를 선택했습니다.")
        if st.button("다회차 예측 고정 및 비교", key="bt_batch_run"):
            try:
                if not chosen:
                    raise ValueError("완료된 회차가 없습니다.")
                items = []
                for rid in chosen:
                    item = prepare_round(frame, product, rid)
                    snap = create_prediction_snapshot(item.context, snapshot_model_input(item.context), config)
                    items.append((snap, item.outcomes))
                evaluations = [evaluate_snapshot(snap, outcomes) for snap, outcomes in items]
                st.session_state["bt_batch"] = {"signature": hashlib.sha256(payload + product.encode() + repr(config).encode() + str(batch_size).encode()).hexdigest(),
                                                  "evaluations": evaluations}
            except ValueError as exc:
                st.error(f"다회차 백테스트 오류: {exc}")
        batch_signature = hashlib.sha256(payload + product.encode() + repr(config).encode() + str(batch_size).encode()).hexdigest()
        batch = st.session_state.get("bt_batch")
        if isinstance(batch, dict) and batch.get("signature") == batch_signature:
            aggregate = aggregate_evaluations(batch["evaluations"])
            show_backtest_metrics(aggregate)
            st.dataframe(aggregate.trend_frame, width="stretch", hide_index=True)
            st.dataframe(aggregate.league_frame, width="stretch", hide_index=True)
            chart = aggregate.trend_frame.sort_values("날짜").set_index("날짜")[["적중률(%)"]]
            st.line_chart(chart)
            detail = st.selectbox("회차별 상세", [item.round_id for item in batch["evaluations"]], key="bt_batch_detail")
            item = next(item for item in batch["evaluations"] if item.round_id == detail)
            st.dataframe(item.match_frame, width="stretch", hide_index=True)
    st.info("배당 수익률은 계산하지 않습니다. 단일 회차 또는 가상 자료로 모델 우수성이나 수익성을 판단할 수 없습니다.")


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

proto_tab, toto_tab, backtest_tab = st.tabs(("프로토 분석", "승무패 14경기", "과거 회차 백테스트"))

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

with backtest_tab:
    show_backtest_tab(config)

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

with st.expander("베트맨 파일 연동 안내"):
    st.markdown(
        f"""
        - 공식 사이트: [{BETMAN_HOME_URL}]({BETMAN_HOME_URL})
        - 이용약관: [{BETMAN_TERMS_URL}]({BETMAN_TERMS_URL})
        - 공개 개발자 API 및 자동 수집 허가 문서는 확인되지 않았습니다.
        - 이용약관 제13조에는 사전 승낙 없이 서비스에서 얻은 정보를 서비스 외
          목적으로 복제·제공하는 행위를 제한하는 내용이 있습니다.

        따라서 앱은 베트맨 사이트를 자동 로그인하거나 스크래핑하지 않습니다.
        사용자가 적법하게 확보한 CSV/XLSX만 회차별로 불러옵니다. 일반 승무패만
        기존 1X2 모델에 연결하며 핸디캡·언더오버·기타 시장은 분석하지 않습니다.
        팀명이 불확실하거나 중복 후보가 있으면 자동 연결하지 않습니다.
        """
    )

st.caption("Football data provided by the Football-Data.org API")
st.caption("Betman round information: user-provided file only (no automated scraping)")

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
