"""검증된 로컬 체크포인트만 모아 2026년 1~30회차 보고서를 생성한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ('승', '무', '패')
ROUND_COLUMNS = ['회차', '확보 경기', 'AI 단식 적중', '홈승 기준선', 'TOP20 최고 적중', 'TOP20 정답 포함', '백테스트 상태', '사유']
SOURCES = [
    ('베트맨 공식 1회차', 'https://www.betman.co.kr/main/mainPage/gamebuy/winrstDetl.do?gmId=G011&gmTs=260001', '직접 화면 확인, 자동 수집 아님'),
    ('베트맨 공식 2회차', 'https://www.betman.co.kr/main/mainPage/gamebuy/winrstDetl.do?gmId=G011&gmTs=260002', '직접 화면 확인, 자동 수집 아님'),
    ('베트맨 공식 3회차', 'https://www.betman.co.kr/main/mainPage/gamebuy/winrstDetl.do?gmId=G011&gmTs=260003', '직접 화면 확인; 4·6번 적중특례'),
    ('football-data.org API', 'https://www.football-data.org/coverage', '경기 ID, 이력 및 결과'),
    ('리그 공식 결과, 피사', 'https://images.legaseriea.it/image/private/fl_attachment/prd/bk75vvzmfnhxockat7v1.pdf', '2025-12-27 결과 확인'),
    ('리그 공식 결과, 크레모네세', 'https://images.legaseriea.it/image/private/fl_attachment/prd/hojzkjmj1pghsvatefgu.pdf', '2025-12-28 결과 확인'),
    ('리그 공식 결과, 사수올로', 'https://img.legaseriea.it/vimages/693850c8/Cu81.pdf', '2025-12-06 결과 확인'),
    ('리그 공식 결과, 번리', 'https://www.premierleague.com/en/news/4522979/burnley-1-newcastle-united-3-match-report-30-december-2025', '2025-12-30 결과 확인'),
    ('리그 공식 결과, 선덜랜드', 'https://www.premierleague.com/en/match/2562057/brighton-and-hove-albion-vs-sunderland', '2025-12-20 결과 확인'),
    ('리그 공식 결과, 리즈', 'https://www.premierleague.com/en/match/2562073/sunderland-vs-leeds-united/stats', '2025-12-28 결과 확인'),
    ('풀럼 경기 연기', 'https://www.standard.co.uk/sport/football/fulham-vs-liverpool-kick-off-delayed-medical-emergency-b1264886.html', '1회차 9번 실제 시작 15분 지연'),
]


def _load_round(directory: Path, number: int):
    paths = [directory / f'round{number:02d}_{suffix}' for suffix in ('prediction_snapshot.json', 'matches.csv', 'top20.csv')]
    if not all(p.is_file() for p in paths):
        return None
    snapshot = json.loads(paths[0].read_text(encoding='utf-8'))
    matches = pd.read_csv(paths[1], encoding='utf-8-sig')
    tops = pd.read_csv(paths[2], encoding='utf-8-sig')
    if len(matches) != 14 or sorted(matches['번호'].tolist()) != list(range(1, 15)) or len(tops) != 20:
        raise ValueError(f'{number}회차 체크포인트의 경기번호 또는 조합이 불완전합니다.')
    if not matches['회차'].eq(number).all() or not tops['회차'].eq(number).all():
        raise ValueError(f'{number}회차 파일의 회차번호가 일치하지 않습니다.')
    predictions = sorted(snapshot['predictions'], key=lambda row: row['number'])
    for match, pred in zip(matches.sort_values('번호').to_dict('records'), predictions, strict=True):
        if (match['번호'], match['홈팀'], match['원정팀'], match['AI 선택']) != (pred['number'], pred['home_team'], pred['away_team'], pred['pick']):
            raise ValueError(f'{number}회차 예측 체크포인트와 결과 파일이 일치하지 않습니다.')
        if match['실제 결과'] not in OUTCOMES or bool(match['적중']) != (match['AI 선택'] == match['실제 결과']):
            raise ValueError(f'{number}회차 적중 결과가 일치하지 않습니다.')
        probs = np.asarray(pred['probabilities'], dtype=float)
        if len(probs) != 3 or np.any(probs < 0) or not np.isclose(probs.sum(), 1) or not np.allclose(probs * 100, [match['홈승%'], match['무%'], match['원정승%']]):
            raise ValueError(f'{number}회차 확률이 예측 체크포인트와 일치하지 않습니다.')
        home_goals, away_goals = map(int, str(match['실제 스코어']).split(':'))
        outcome = '승' if home_goals > away_goals else '무' if home_goals == away_goals else '패'
        if outcome != match['실제 결과']:
            raise ValueError(f'{number}회차 실제 스코어와 결과가 일치하지 않습니다.')
        actual = OUTCOMES.index(outcome)
        brier = sum((p - (i == actual)) ** 2 for i, p in enumerate(probs))
        log_loss = -np.log(max(probs[actual], 1e-15))
        if not np.isclose(match['Brier Score'], brier) or not np.isclose(match['Log Loss'], log_loss):
            raise ValueError(f'{number}회차 확률 성능 지표가 일치하지 않습니다.')
    actual_results = matches.sort_values('번호')['실제 결과'].tolist()
    for row, pred in zip(tops.sort_values('순위').to_dict('records'), snapshot['top20'], strict=True):
        if row['조합'] != '-'.join(pred['outcomes']) or not np.isclose(row['확률'], pred['probability']) or row['일치 경기수'] != sum(a == b for a, b in zip(actual_results, pred['outcomes'], strict=True)):
            raise ValueError(f'{number}회차 TOP20 체크포인트와 일치하지 않습니다.')
    return snapshot, matches, tops


def generate(input_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds, all_matches, all_tops, errors = [], [], [], []
    freq_briers, freq_logs = [], []
    for number in range(1, 31):
        try:
            loaded = _load_round(input_dir, number)
            if loaded is None:
                raise ValueError('공식 경기표 및 경기 전 이력 대조·예측 체크포인트 미완료')
            snap, matches, tops = loaded
            best = int(tops['일치 경기수'].max())
            matches['기준시각'] = snap['cutoff']
            matches['모델 버전'] = snap['model_version']
            matches['사용 이력 범위'] = snap['source_scope']
            for _, row in matches.iterrows():
                frequencies = snap['baseline_frequencies'][row['리그']]
                p = [float(frequencies[o]) for o in OUTCOMES]
                actual = OUTCOMES.index(row['실제 결과'])
                freq_briers.append(sum((x - (i == actual)) ** 2 for i, x in enumerate(p)))
                freq_logs.append(-np.log(max(p[actual], 1e-15)))
            rounds.append([number, '14/14', f"{int(matches['적중'].sum())}/14", f"{int(matches['홈승 기준선 적중'].sum())}/14", f'{best}/14', '예' if best == 14 else '아니오', '완료 · 이력 제한 큼', ''])
            all_matches.append(matches)
            all_tops.append(tops)
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            reason = str(exc)
            observed = input_dir / f'round{number:02d}_observation.json'
            secured = '미검증'
            if observed.is_file():
                try:
                    observation = json.loads(observed.read_text(encoding='utf-8'))
                    if observation.get('round') == number and observation.get('official_fixture_count') == 14:
                        secured = '14/14 · 공식 화면 확인'
                        reason = str(observation.get('reason') or reason)
                except (ValueError, OSError, json.JSONDecodeError):
                    pass
            rounds.append([number, secured, '', '', '', '', '미완료', reason])
            errors.append({'회차':number, '사유':reason})
    rd = pd.DataFrame(rounds, columns=ROUND_COLUMNS)
    mt = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame(columns=['회차','번호','홈팀','원정팀','홈승%','무%','원정승%','AI 선택','실제 결과','적중'])
    tt = pd.concat(all_tops, ignore_index=True) if all_tops else pd.DataFrame(columns=['회차','순위','조합','확률','일치 경기수'])
    er = pd.DataFrame(errors, columns=['회차','사유'])
    sources = pd.DataFrame(SOURCES, columns=['출처', 'URL', '검증 내용'])
    results = {'rounds':rd, 'matches':mt, 'top20':tt, 'errors':er, 'sources':sources}
    for name, frame in results.items():
        frame.to_csv(output_dir / f'backtest_2026_1_30_{name}.csv', index=False, encoding='utf-8-sig')
    lines = ['# 2026 승무패 1~30회차 과거 시점 재현', '',
             '실제 당시 저장된 예측이 아닌 사후 재구성이다.',
             f'완료: {len(all_matches)}/30회차, {len(mt)}경기. 미완료 회차는 성능 통계에서 제외.', '']
    if not mt.empty:
        lines += ['2024/25 과거 이력과 공식 검증한 소수 경기만 사용했다. 같은 시즌의 다수 경기와 당시 자료 가용성은 검증되지 않아 제외했다.',
                  '승격팀에는 1~2경기만 있는 등 모든 경기의 전력 추정에 높은 불확실성이 있다.',
                  '1회차 9번: 베트맨 예정시각 00:00, API 00:15. 당시 기사에서 15분 지연을 확인했다. 두 시각 모두 회차 마감 이후다.', '']
        counts = mt.groupby('회차')['적중'].sum()
        home = mt['홈승 기준선 적중'].astype(bool)
        y = mt['실제 결과']
        home_brier = ((y != '승') * 2).mean()
        home_loss = ((y != '승') * -np.log(1e-15)).mean()
        lines += [f"AI: {int(mt['적중'].sum())}/{len(mt)} ({mt['적중'].mean()*100:.2f}%), 회차 평균 {counts.mean():.2f}/14",
                  f"홈승 기준선: {int(home.sum())}/{len(mt)} ({home.mean()*100:.2f}%)",
                  f"과거 빈도 기준선: {int(mt['빈도 기준선 적중'].sum())}/{len(mt)} ({mt['빈도 기준선 적중'].mean()*100:.2f}%)",
                  f"AI Brier {mt['Brier Score'].mean():.6f} / Log Loss {mt['Log Loss'].mean():.6f}",
                  f"홈승 결정적 확률 기준 Brier {home_brier:.6f} / Log Loss {home_loss:.6f} (0 확률은 1e-15로 제한)",
                  f"과거 빈도 확률 기준 Brier {np.mean(freq_briers):.6f} / Log Loss {np.mean(freq_logs):.6f}",
                  f"TOP20 정답 포함 {int((tt.groupby('회차')['일치 경기수'].max()==14).sum())}회차; 최고 일치 {int(tt['일치 경기수'].max())}/14",
                  *(f'단식 {k}개 이상: {int((counts >= k).sum())}회차' for k in (11,12,13,14)), '']
        for outcome in OUTCOMES:
            group = mt[mt['실제 결과']==outcome]
            lines.append(f"실제 {outcome}: {len(group)}경기 중 AI {outcome} 적중 {int(group['적중'].sum())}경기")
    lines += ['', '## 회차별 현황', '', '| 회차 | 확보 | AI | 홈승 | TOP20 최고 | 정답 포함 | 상태 | 사유 |', '|---|---|---|---|---|---|---|---|']
    for row in rounds:
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    (output_dir / 'backtest_2026_1_30_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('reports/local_backtest_2026'))
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    tables = generate(args.input, args.output)
    print(f"분석 완료: {tables['rounds']['백테스트 상태'].str.startswith('완료').sum()}/30회차, {len(tables['matches'])}경기")


if __name__ == '__main__':
    main()
