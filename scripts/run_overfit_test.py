"""
3차 MVP: 오버핏 진단 실행
==========================

2차 MVP의 4개 페어 + GDX~KO에 대해:
  1. 파라미터 민감도 분석 (9개 (window, entry) 조합 × 5-fold)
  2. Permutation Test (200회 셔플)
  3. 종합 등급 (A~F)
  4. 시각화: heatmap + null distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder, Pair
from purged_kfold import PurgedKFold
from overfit_diagnostics import (
    ParameterSensitivityAnalyzer,
    PermutationTester,
    diagnose_pair,
)


# ============================================================
# 1. 데이터 + 페어 풀 재구성
# ============================================================
TICKERS = [
    'KO', 'PEP', 'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX', 'V', 'MA', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T', 'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW', 'WMT', 'TGT',
]

print('=' * 70)
print('1. 데이터 + 페어 풀 재구성')
print('=' * 70)
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)
pairs = finder.screen_pairs(data)

# GDX~KO도 강제 추가 (1차 MVP에서 발굴, 비교 목적)
train_cut = int(len(data) * 0.7)
train_data = data.iloc[:train_cut]
gdx_ko_beta = finder.tls_beta(
    np.log(train_data['KO']).values,
    np.log(train_data['GDX']).values,
)
spread = np.log(train_data['GDX']) - gdx_ko_beta * np.log(train_data['KO'])
gdx_ko_hl = finder.half_life(spread) or 14.7
gdx_ko_pair = Pair(y='GDX', x='KO', beta=gdx_ko_beta,
                   pvalue=0.05, half_life=gdx_ko_hl)
pairs_to_test = pairs + [gdx_ko_pair]

print(f'  진단 대상 페어: {len(pairs_to_test)}개')
for i, p in enumerate(pairs_to_test, 1):
    print(f'    {i}. {p}')


# ============================================================
# 2. 진단 도구 초기화
# ============================================================
print()
print('=' * 70)
print('2. 진단 도구 초기화')
print('=' * 70)

kfold = PurgedKFold(n_splits=5, purge_days=30, embargo_pct=0.01)
analyzer = ParameterSensitivityAnalyzer(
    windows=[20, 30, 40],
    entries=[1.5, 2.0, 2.5],
    exits=[0.5],
    stop=3.5,
    kfold=kfold,
)
tester = PermutationTester(
    n_permutations=200,
    sig_window=30,
    entry=2.0,
    exit_thr=0.5,
    stop=3.5,
    random_state=42,
)
print('  파라미터 격자: 3 windows × 3 entries = 9 조합')
print('  Permutation: 200회 셔플')


# ============================================================
# 3. 진단 실행
# ============================================================
print()
print('=' * 70)
print('3. 진단 실행')
print('=' * 70)

reports = []
for i, p in enumerate(pairs_to_test, 1):
    print(f'  [{i}/{len(pairs_to_test)}] {p.y}~{p.x} ... ', end='', flush=True)
    try:
        r = diagnose_pair(data[p.y], data[p.x], p, analyzer, tester)
        reports.append(r)
        print(f'Grade {r.overall_grade}')
    except Exception as e:
        print(f'FAIL ({e})')


# ============================================================
# 4. 종합 결과
# ============================================================
print()
print('=' * 70)
print('4. 종합 결과 (Grade 순)')
print('=' * 70)
grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
reports.sort(key=lambda r: grade_order.get(r.overall_grade, 5))

print()
print(f'  {"Pair":<14s} {"Grade":>6s} '
      f'{"Sens μ":>9s} {"Sens σ":>9s} {"+%셀":>7s} '
      f'{"Perm p":>8s} {"Perm z":>8s}')
for r in reports:
    print(f'  {r.pair.y+"~"+r.pair.x:<14s} {r.overall_grade:>6s} '
          f'{r.sensitivity.mean_sharpe:>+9.2f} '
          f'{r.sensitivity.std_sharpe:>9.2f} '
          f'{r.sensitivity.positive_pct:>7.0%} '
          f'{r.permutation.p_value:>8.3f} '
          f'{r.permutation.z_score:>+8.2f}')


# ============================================================
# 5. 시각화
# ============================================================
print()
print('=' * 70)
print('5. 시각화')
print('=' * 70)

n_pairs = len(reports)
fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))
if n_pairs == 1:
    axes = axes.reshape(2, 1)

for col, r in enumerate(reports):
    # 상단: Heatmap of Sharpe across (window, entry)
    ax = axes[0, col]
    grid = r.sensitivity.sharpe_grid.astype(float)
    im = ax.imshow(grid.values, cmap='RdYlGn', aspect='auto',
                   vmin=-2, vmax=2)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f'{c:.1f}' for c in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel('Entry threshold')
    ax.set_ylabel('Window')
    ax.set_title(f'{r.pair.y}~{r.pair.x}  [{r.overall_grade}]\n'
                 f'Sens μ={r.sensitivity.mean_sharpe:+.2f} '
                 f'σ={r.sensitivity.std_sharpe:.2f}',
                 fontsize=10)
    # 셀에 값 표시
    for ri in range(len(grid.index)):
        for ci in range(len(grid.columns)):
            val = grid.values[ri, ci]
            if not np.isnan(val):
                ax.text(ci, ri, f'{val:.1f}', ha='center', va='center',
                        fontsize=9,
                        color='white' if abs(val) > 1.2 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 하단: Permutation null distribution
    ax = axes[1, col]
    null_dist = r.permutation.null_distribution
    ax.hist(null_dist, bins=30, alpha=0.6, color='gray',
            edgecolor='black', label='Null (shuffled)')
    ax.axvline(r.permutation.real_sharpe, color='red', lw=2,
               label=f'Real Sharpe={r.permutation.real_sharpe:+.2f}')
    ax.axvline(r.permutation.null_mean, color='blue', ls='--', alpha=0.5,
               label=f'Null μ={r.permutation.null_mean:+.2f}')
    ax.set_xlabel('Sharpe ratio')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Permutation Test (B=200)\n'
                 f'p={r.permutation.p_value:.3f}, '
                 f'z={r.permutation.z_score:+.2f}',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = '/home/claude/overfit_diagnostics_chart.png'
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'  차트 저장: {out_path}')


# ============================================================
# 6. 등급별 해석
# ============================================================
print()
print('=' * 70)
print('6. 등급별 해석')
print('=' * 70)

grade_groups = {}
for r in reports:
    grade_groups.setdefault(r.overall_grade, []).append(r)

interp = {
    'A': '✓ 진짜 견고 — 파라미터 무관 + 통계적 유의. 실전 운용 후보.',
    'B': '✓ 견고 — 1개 기준만 미달. 모니터링하며 운용 가능.',
    'C': '△ 경계 — 절반만 통과. 신중한 추가 검증 필요.',
    'D': '✗ 위험 — 1개 기준만 통과. 오버핏 가능성 높음.',
    'F': '✗ 오버핏 의심 — 모든 기준 실패. 운용 금지.',
}
for grade in ['A', 'B', 'C', 'D', 'F']:
    rs = grade_groups.get(grade, [])
    print(f'\n  [{grade}] {interp[grade]}')
    for r in rs:
        print(f'    - {r.summary()}')


# ============================================================
# 7. 1차 MVP / 2차 MVP / 3차 MVP 비교 — GDX~KO
# ============================================================
gdx_ko_report = next((r for r in reports
                      if r.pair.y == 'GDX' and r.pair.x == 'KO'), None)
if gdx_ko_report:
    print()
    print('=' * 70)
    print('7. GDX~KO 진단 변천사')
    print('=' * 70)
    print()
    print('  1차 MVP (단일 split):  Sharpe -0.49, 수익률 -3.28% → "망함"')
    print('  2차 MVP (5-fold):       Sharpe +2.24 ± 2.05, 4/5 fold 양수 → "견고"')
    print(f'  3차 MVP (오버핏 진단):  Grade {gdx_ko_report.overall_grade}')
    print(f'    - 파라미터 민감도: {gdx_ko_report.sensitivity.mean_sharpe:+.2f} '
          f'± {gdx_ko_report.sensitivity.std_sharpe:.2f}, '
          f'양수 셀 {gdx_ko_report.sensitivity.positive_pct:.0%}')
    print(f'    - Permutation:   p={gdx_ko_report.permutation.p_value:.3f}, '
          f'z={gdx_ko_report.permutation.z_score:+.2f}')
    if gdx_ko_report.overall_grade in ['A', 'B']:
        print(f'    → 진짜 견고한 페어. 2차 MVP의 판정이 옳았음.')
    else:
        print(f'    → 오버핏 의심. 2차 MVP의 +1.09도 fluke였을 가능성.')
