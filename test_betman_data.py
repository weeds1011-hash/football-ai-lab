from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from betman_data import (
    MARKET_1X2,
    MARKET_HANDICAP,
    MARKET_OTHER,
    MARKET_TOTALS,
    available_rounds,
    classify_market,
    latest_retrieved_at,
    link_betman_matches,
    normalise_betman_dataframe,
    read_betman_file,
    read_team_alias_file,
    select_round,
    validate_toto_round,
)
from models import analyze_dataframe


def _raw_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "상품": "프로토 승부식",
        "회차": "127회차",
        "경기번호": "1경기",
        "종목": "축구",
        "대회": "가상 프리미어리그",
        "대회코드": "PL",
        "경기시각(KST)": "2026-10-31 23:00",
        "마감시각(KST)": "2026-10-31 22:50",
        "홈팀": "맨시티",
        "원정팀": "리버풀",
        "시장유형": "승무패",
        "조회기준시각(KST)": "2026-10-27 16:42",
        "출처 URL": "https://www.betman.co.kr/",
    }
    row.update(updates)
    return row


def _fixture_frame(*, duplicate: bool = False) -> pd.DataFrame:
    row = {
        "경기번호": "9001",
        "경기시각(KST)": "2026-10-31 23:00",
        "대회코드": "PL",
        "홈팀 ID": 65,
        "원정팀 ID": 64,
        "홈팀": "Manchester City FC",
        "원정팀": "Liverpool FC",
        "홈 기대득점": 1.7,
        "원정 기대득점": 1.2,
        "홈 Elo": 1640.0,
        "원정 Elo": 1605.0,
    }
    rows = [row]
    if duplicate:
        rows.append({**row, "경기번호": "9002"})
    return pd.DataFrame(rows)


def _alias_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition_code": ["PL", "PL"],
            "source_team": ["맨시티", "리버풀"],
            "target_team": ["Manchester City FC", "Liverpool FC"],
        }
    )


def test_rounds_and_match_numbers_keep_official_order() -> None:
    raw = pd.DataFrame(
        [
            _raw_row(**{"회차": "128회차", "경기번호": "2경기"}),
            _raw_row(**{"회차": "127회차", "경기번호": "2경기"}),
            _raw_row(**{"회차": "127회차", "경기번호": "1경기"}),
        ]
    )
    normalised = normalise_betman_dataframe(raw)
    assert available_rounds(normalised) == ["127회차", "128회차"]
    assert select_round(normalised, "127회차")["match_no"].tolist() == [1, 2]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("승무패", MARKET_1X2),
        ("일반 1X2", MARKET_1X2),
        ("핸디캡 H -1.0", MARKET_HANDICAP),
        ("언더오버 U/O 2.5", MARKET_TOTALS),
        ("SUM 홀짝", MARKET_OTHER),
    ],
)
def test_market_types_are_not_confused(label: str, expected: str) -> None:
    assert classify_market(label) == expected


def test_csv_cp949_and_xlsx_uploads_are_supported() -> None:
    raw = pd.DataFrame([_raw_row()])
    csv_payload = raw.to_csv(index=False).encode("cp949")
    assert len(read_betman_file(csv_payload, "round.csv")) == 1

    output = BytesIO()
    raw.to_excel(output, index=False, engine="openpyxl")
    assert len(read_betman_file(output.getvalue(), "round.xlsx")) == 1


def test_timezone_and_retrieved_at_are_kst() -> None:
    frame = normalise_betman_dataframe(pd.DataFrame([_raw_row()]))
    assert str(frame.loc[0, "kickoff_kst"].tzinfo) == "Asia/Seoul"
    assert latest_retrieved_at(frame) == "2026-10-27 16:42"


def test_toto_round_requires_exact_1_to_14_football_matches() -> None:
    rows = [
        _raw_row(
            **{
                "상품": "축구토토 승무패",
                "회차": "64회차",
                "경기번호": f"{number}경기",
                "홈팀": f"가상 홈 {number}",
                "원정팀": f"가상 원정 {number}",
            }
        )
        for number in range(1, 15)
    ]
    frame = normalise_betman_dataframe(pd.DataFrame(rows))
    validate_toto_round(frame)

    with pytest.raises(ValueError, match="정확히 14경기"):
        validate_toto_round(frame.iloc[:-1])

    wrong_order = frame.copy()
    wrong_order.loc[13, "match_no"] = 15
    with pytest.raises(ValueError, match="공식 순서"):
        validate_toto_round(wrong_order)


def test_explicit_alias_mapping_connects_match_and_keeps_missing_odds_empty() -> None:
    betman = normalise_betman_dataframe(pd.DataFrame([_raw_row()]))
    result = link_betman_matches(
        betman,
        _fixture_frame(),
        alias_frame=_alias_frame(),
        supported_codes={"PL"},
    )
    assert result.matched_count == 1
    assert result.status_frame.loc[0, "매칭 상태"] == "자동 매칭 완료"
    assert pd.isna(result.model_frame.loc[0, "홈승 배당"])

    _, output, _ = analyze_dataframe(result.model_frame)
    assert pd.isna(output.loc[0, "홈승 기대수익률(%)"])
    probability_columns = ["홈승 확률(%)", "무승부 확률(%)", "원정승 확률(%)"]
    assert output.loc[0, probability_columns].sum() == pytest.approx(100.0)


def test_similar_name_is_not_automatically_matched() -> None:
    betman = normalise_betman_dataframe(
        pd.DataFrame([_raw_row(**{"홈팀": "Manchester", "원정팀": "Liverpool FC"})])
    )
    result = link_betman_matches(
        betman,
        _fixture_frame(),
        supported_codes={"PL"},
    )
    assert result.model_frame.empty
    assert result.status_frame.loc[0, "매칭 상태"] == "매칭 실패"


def test_team_ids_take_precedence_over_matching_names() -> None:
    betman = normalise_betman_dataframe(
        pd.DataFrame(
            [
                _raw_row(
                    **{
                        "홈팀": "Manchester City FC",
                        "원정팀": "Liverpool FC",
                        "홈팀 ID": 999,
                        "원정팀 ID": 64,
                    }
                )
            ]
        )
    )
    result = link_betman_matches(
        betman,
        _fixture_frame(),
        supported_codes={"PL"},
    )
    assert result.model_frame.empty


def test_duplicate_candidates_require_user_confirmation() -> None:
    betman = normalise_betman_dataframe(pd.DataFrame([_raw_row()]))
    result = link_betman_matches(
        betman,
        _fixture_frame(duplicate=True),
        alias_frame=_alias_frame(),
        supported_codes={"PL"},
    )
    assert result.model_frame.empty
    assert result.status_frame.loc[0, "매칭 상태"] == "사용자 확인 필요"


def test_unsupported_competition_and_non_1x2_are_not_modelled() -> None:
    unsupported = normalise_betman_dataframe(
        pd.DataFrame([_raw_row(**{"대회코드": "MLS"})])
    )
    unsupported_result = link_betman_matches(
        unsupported,
        pd.DataFrame(),
        supported_codes={"PL"},
    )
    assert unsupported_result.status_frame.loc[0, "매칭 상태"] == "전력 데이터 부족"

    handicap = normalise_betman_dataframe(
        pd.DataFrame([_raw_row(**{"시장유형": "핸디캡 H -1.0"})])
    )
    handicap_result = link_betman_matches(
        handicap,
        _fixture_frame(),
        alias_frame=_alias_frame(),
        supported_codes={"PL"},
    )
    assert handicap_result.model_frame.empty
    assert handicap_result.status_frame.loc[0, "매칭 상태"] == "분석 불가 시장"


def test_alias_file_structure_and_changed_input_structure_errors() -> None:
    alias_payload = pd.DataFrame(
        {
            "대회코드": ["PL"],
            "베트맨 팀명": ["맨시티"],
            "football-data 팀명": ["Manchester City FC"],
        }
    ).to_csv(index=False).encode("utf-8-sig")
    aliases = read_team_alias_file(alias_payload, "aliases.csv")
    assert aliases.loc[0, "target_team"] == "Manchester City FC"

    broken = pd.DataFrame({"회차": ["1회차"], "경기번호": [1]})
    with pytest.raises(ValueError, match="필수 열"):
        normalise_betman_dataframe(broken)


def test_conflicting_aliases_are_rejected_instead_of_guessed() -> None:
    betman = normalise_betman_dataframe(pd.DataFrame([_raw_row()]))
    ambiguous_aliases = pd.DataFrame(
        {
            "competition_code": ["PL", "PL", "PL"],
            "source_team": ["맨시티", "맨시티", "리버풀"],
            "target_team": ["Manchester City FC", "Manchester United FC", "Liverpool FC"],
        }
    )
    with pytest.raises(ValueError, match="서로 다른"):
        link_betman_matches(
            betman,
            _fixture_frame(),
            alias_frame=ambiguous_aliases,
            supported_codes={"PL"},
        )


def test_match_time_outside_tolerance_is_not_connected() -> None:
    betman = normalise_betman_dataframe(
        pd.DataFrame([_raw_row(**{"경기시각(KST)": "2026-11-02 23:00"})])
    )
    result = link_betman_matches(
        betman,
        _fixture_frame(),
        alias_frame=_alias_frame(),
        supported_codes={"PL"},
    )
    assert result.model_frame.empty
    assert result.status_frame.loc[0, "매칭 상태"] == "매칭 실패"
