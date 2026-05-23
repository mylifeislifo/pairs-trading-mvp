"""
2차 MVP: K-fold 검증으로 페어 견고성 비교
==========================================

1차 MVP의 한계: 단일 70/30 split → IS Sharpe 1.64가 OOS에서 -0.49로 무너짐.
2차 MVP의 목표: 5-fold로 GDX~KO가 진짜 좋은 페어인지, 우연한 fluke인지 판단.

핵심 가설:
  H1. 진짜 견고한 페어는 모든 fold에서 어느 정도 양의 수익을 냄
  H2. fluke 페어는 fold마다 들쭉날쭉, 평균은 0 근처
  H3. 견고성 점수 (mean_sharpe / std_sharpe)가 진짜 신호
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder, compute_spread
from purged_kfold import PurgedKFold, kfold_evaluate_pair


# ============================================================
# 1. 데이터 로드 (1차 MVP와 동일)
# ============================================================
TICKERS = [
    'KO', 'PEP',
    'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX',
    'V', 'MA',
    'AAPL', 'MSFT',
    'NVDA', 'AMD',
    'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T',
    'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW',
    'WMT', 'TGT',
]

print('=' * 70)
print('1. 데이터 다운로드')
print('=' * 70)
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'  {data.shape[0]}일 x {data.shape[1]}종목, '
      f'{data.index[0].date()} ~ {data.index[-1].date()}')


# ============================================================
# 2. 페어 발굴 (전체 데이터 사용 — 발굴 단계는 leakage 인정)
#   진짜 엄격하려면 매 fold에서 train 부분만으로 재발굴해야 하지만,
#   페어 비교가 어려워지므로 MVP에선 페어 후보는 고정.
# ============================================================
print()
print('=' * 70)
print('2. 페어 발굴 (전체 기간) — 후보 풀 고정')
print('=' * 70)
finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)
pairs = finder.screen_pairs(data)
print(f'  적격 페어: {len(pairs)}개')
for i, p in enumerate(pairs, 1):
    print(f'  {i}. {p}')


# ============================================================
# 3. K-fold 검증
# ============================================================
print()
print('=' * 70)
print('3. Purged K-fold 검증 (n_splits=5, purge=30d, embargo=1%)')
print('=' * 70)

kfold = PurgedKFold(n_splits=5, purge_days=30, embargo_pct=0.01)

reports = []
for p in pairs:
    sig_window = max(20, min(60, int(p.half_life * 2)))
    report = kfold_evaluate_pair(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window,
        entry=2.0, exit_thr=0.5, stop=3.5,
        kelly_fraction=0.25, kelly_cap=0.20,
    )
    reports.append(report)

# 견고성 점수 순으로 정렬
reports.sort(key=lambda r: r.robustness_score, reverse=True)

print()
print('  견고성 순위 (Robustness = mean_sharpe / std_sharpe):')
print(f'  {"#":<3s} {"Pair":<18s} {"평균수익":>10s} {"수익σ":>10s} '
      f'{"Sharpe μ":>10s} {"Sharpe σ":>10s} {"+Fold":>8s} {"견고성":>10s}')
for i, r in enumerate(reports, 1):
    if r.total_folds == 0:
        print(f'  {i:<3d} {r.pair.y+"~"+r.pair.x:<18s}   (fold 평가 불가)')
        continue
    print(f'  {i:<3d} {r.pair.y+"~"+r.pair.x:<18s} '
          f'{r.mean_val_return:>+10.2%} '
          f'{r.std_val_return:>10.2%} '
          f'{r.mean_val_sharpe:>+10.2f} '
          f'{r.std_val_sharpe:>10.2f} '
          f'{r.profitable_folds:>3d}/{r.total_folds:<3d} '
          f'{r.robustness_score:>+10.2f}')


# ============================================================
# 4. Fold별 상세 분포 (상위 4개 페어)
# ============================================================
print()
print('=' * 70)
print('4. Fold별 OOS 수익률 분포 (상위 4개)')
print('=' * 70)
for r in reports[:4]:
    if r.total_folds == 0:
        continue
    print(f'\n  {r.pair.y}~{r.pair.x}:')
    print(f'    {"Fold":<6s} {"Kelly":>8s} '
          f'{"Train거래":>10s} {"Train승률":>10s} '
          f'{"Val거래":>10s} {"Val승률":>10s} {"Val수익":>10s} {"Val Sharpe":>11s}')
    for f in r.fold_results:
        print(f'    {f.fold_idx:<6d} {f.kelly_f:>8.2%} '
              f'{f.train_n_trades:>10d} {f.train_win_rate:>10.2%} '
              f'{f.val_n_trades:>10d} {f.val_win_rate:>10.2%} '
              f'{f.val_total_return:>+10.2%} {f.val_sharpe:>+11.2f}')


# ============================================================
# 5. 1차 MVP 단일 split 결과와 비교 — GDX~KO 강제 평가
# ============================================================
print()
print('=' * 70)
print('5. 단일 split (1차 MVP) vs K-fold (2차 MVP) 비교')
print('=' * 70)

# GDX~KO는 전체 기간 공적분 검정에서 떨어졌지만, 1차 MVP와 직접 비교를 위해
# Train 70% 구간의 베타로 강제 K-fold 평가
from pairs_trading_mvp import Pair
train_cut = int(len(data) * 0.7)
train_data = data.iloc[:train_cut]
gdx_ko_beta = finder.tls_beta(
    np.log(train_data['KO']).values,
    np.log(train_data['GDX']).values,
)
# 베타가 비합리적이면 OLS 폴백
if not (0.5 < abs(gdx_ko_beta) < 5):
    gdx_ko_beta = float(np.polyfit(np.log(train_data['KO']).values,
                                    np.log(train_data['GDX']).values, 1)[0])
spread_train = (np.log(train_data['GDX']) - gdx_ko_beta * np.log(train_data['KO']))
gdx_ko_hl = finder.half_life(spread_train) or 14.7

gdx_ko_pair = Pair(y='GDX', x='KO', beta=gdx_ko_beta,
                   pvalue=0.05, half_life=gdx_ko_hl)
print(f'  강제 평가 페어: {gdx_ko_pair}')

gdx_ko_report = kfold_evaluate_pair(
    data['GDX'], data['KO'], gdx_ko_pair, kfold,
    sig_window=max(20, min(60, int(gdx_ko_hl * 2))),
    entry=2.0, exit_thr=0.5, stop=3.5,
)

print()
print('  GDX~KO:')
print(f'    1차 MVP (단일 70/30 split):')
print(f'      IS Sharpe : +1.64')
print(f'      OOS Sharpe: -0.49 (단일 fold)')
print(f'      OOS 수익률: -3.28%')
if gdx_ko_report.total_folds > 0:
    print(f'    2차 MVP (5-fold):')
    print(f'      평균 OOS Sharpe : {gdx_ko_report.mean_val_sharpe:+.2f} '
          f'± {gdx_ko_report.std_val_sharpe:.2f}')
    print(f'      평균 OOS 수익률 : {gdx_ko_report.mean_val_return:+.2%} '
          f'± {gdx_ko_report.std_val_return:.2%}')
    print(f'      수익 fold       : {gdx_ko_report.profitable_folds}/'
          f'{gdx_ko_report.total_folds}')
    print(f'      견고성 점수      : {gdx_ko_report.robustness_score:+.2f}')
    print()
    print('  Fold 별:')
    for f in gdx_ko_report.fold_results:
        print(f'    Fold {f.fold_idx}: Train Sharpe={f.train_sharpe:+.2f}, '
              f'Val Sharpe={f.val_sharpe:+.2f}, '
              f'Val Return={f.val_total_return:+.2%}, '
              f'Kelly={f.kelly_f:.1%}')
    print()
    print('  → 5-fold가 단일 split의 -3.28% 결과를 어떻게 진단하는지가 핵심.')


# ============================================================
# 6. 시각화
# ============================================================
print()
print('=' * 70)
print('6. 시각화')
print('=' * 70)

valid_reports = [r for r in reports if r.total_folds > 0]

if valid_reports:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) 페어별 fold 수익률 boxplot
    ax = axes[0, 0]
    box_data = []
    box_labels = []
    for r in valid_reports:
        returns = [f.val_total_return for f in r.fold_results]
        box_data.append(returns)
        box_labels.append(f'{r.pair.y}~{r.pair.x}')
    bp = ax.boxplot(box_data, labels=box_labels, showmeans=True,
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_ylabel('OOS Return per fold')
    ax.set_title('Fold-wise Return Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

    # (b) 페어별 견고성 점수 막대
    ax = axes[0, 1]
    pair_names = [f'{r.pair.y}~{r.pair.x}' for r in valid_reports]
    scores = [r.robustness_score for r in valid_reports]
    colors = ['darkgreen' if s > 0 else 'darkred' for s in scores]
    bars = ax.barh(pair_names, scores, color=colors, alpha=0.7)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Robustness (mean_Sharpe / std_Sharpe)')
    ax.set_title('Pair Robustness Score (higher = more consistent)')
    ax.grid(alpha=0.3, axis='x')
    for bar, score in zip(bars, scores):
        ax.text(score, bar.get_y() + bar.get_height() / 2,
                f' {score:+.2f}', va='center',
                ha='left' if score > 0 else 'right', fontsize=9)

    # (c) 평균 OOS 수익률 ± std (error bar)
    ax = axes[1, 0]
    means = [r.mean_val_return for r in valid_reports]
    stds = [r.std_val_return for r in valid_reports]
    x_pos = np.arange(len(valid_reports))
    ax.errorbar(x_pos, means, yerr=stds, fmt='o',
                markersize=8, capsize=5, color='steelblue',
                ecolor='gray', elinewidth=2)
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_names, rotation=45, ha='right')
    ax.set_ylabel('Mean OOS Return ± 1σ')
    ax.set_title('OOS Return: Mean ± 1 Standard Deviation')
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

    # (d) 진짜 견고한 페어 vs fluke 비교 (Train Sharpe vs Val Sharpe scatter)
    ax = axes[1, 1]
    for r in valid_reports:
        train_sharpes = [f.train_sharpe for f in r.fold_results]
        val_sharpes = [f.val_sharpe for f in r.fold_results]
        ax.scatter(train_sharpes, val_sharpes, label=f'{r.pair.y}~{r.pair.x}',
                   s=50, alpha=0.6)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    # y=x 라인 (이상적인 경우)
    lims = [-3, 3]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x (ideal)')
    ax.set_xlabel('Train Sharpe (per fold)')
    ax.set_ylabel('Val Sharpe (per fold)')
    ax.set_title('Train vs Val Sharpe — Generalization Check')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    out_path = '/home/claude/kfold_robustness_chart.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {out_path}')

# ============================================================
# 7. 결론 보고서
# ============================================================
print()
print('=' * 70)
print('7. 결론')
print('=' * 70)

robust_pairs = [r for r in valid_reports
                if r.robustness_score > 0.5
                and r.profitable_folds >= r.total_folds * 0.6]
print(f'  진짜 견고한 페어 (견고성>0.5 + 수익fold>=60%): {len(robust_pairs)}개')
for r in robust_pairs:
    print(f'    ✓ {r.summary()}')

fluke_pairs = [r for r in valid_reports
               if r.robustness_score < 0
               or r.profitable_folds < r.total_folds * 0.5]
print(f'\n  Fluke / 위험 페어: {len(fluke_pairs)}개')
for r in fluke_pairs:
    print(f'    ✗ {r.summary()}')
