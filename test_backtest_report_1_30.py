from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest_report_1_30 import generate


def test_no_validated_rounds_reports_thirty_missing(tmp_path: Path) -> None:
    result = generate(tmp_path / 'input', tmp_path / 'output')
    assert result['rounds']['회차'].tolist() == list(range(1, 31))
    assert len(result['errors']) == 30
    assert result['matches'].empty
    assert not result['rounds']['백테스트 상태'].str.startswith('완료').any()


def test_corrupt_round_excluded_and_31_out_of_scope(tmp_path: Path) -> None:
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'round01_prediction_snapshot.json').write_text('{}', encoding='utf-8')
    pd.DataFrame({'회차': [1], '번호': [1]}).to_csv(source / 'round01_matches.csv', index=False)
    pd.DataFrame({'회차': [1]}).to_csv(source / 'round01_top20.csv', index=False)
    (source / 'round31_prediction_snapshot.json').write_text('{}', encoding='utf-8')
    result = generate(source, tmp_path / 'output')
    assert result['rounds'].iloc[0]['백테스트 상태'] == '미완료'
    assert '불완전' in result['rounds'].iloc[0]['사유']
    assert 31 not in result['rounds']['회차'].tolist()
    assert result['matches'].empty


def test_missing_checkpoint_does_not_evaluate_results(tmp_path: Path) -> None:
    source = tmp_path / 'input'
    source.mkdir()
    pd.DataFrame({'회차': [1] * 14, '번호': list(range(1, 15))}).to_csv(source / 'round01_matches.csv', index=False)
    result = generate(source, tmp_path / 'output')
    assert result['matches'].empty
    assert result['rounds'].iloc[0]['AI 단식 적중'] == ''
