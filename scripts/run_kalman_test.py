"""
4차 MVP: 정적 베타 vs Kalman 동적 베타 직접 비교
==================================================

핵심 가설:
  "Kalman 동적 베타가 GDX~KO의 fold 3 (regime change 구간) 손실을 줄여준다."

실험:
  1. 같은 페어 풀에 정적/동적 두 방식 모두 적용
  2. K-fold 결과 비교 (특히 fold별 분포)
  3. GDX~KO에 대해 spread + 베타 + fold별 상세 비교
  4. 결과가 어느 쪽이든 다음 단계 명확:
     - 개선 → Kalman을 표준 도구로 채택, 다른 페어에도 적용
     - 비개선 → Kalman만으로는 부족, 롤링 재발굴이 정답
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder, Pair, compute_spread
from purged_kfold import PurgedKFold, kfold_evaluate_pair
from kalman_pairs import (
    kalman_dynamic_beta, compute_dynamic_spread,
    kfold_evaluate_pair_dynamic,
)


# ============================================================
# 1. 데이터 + 페어 풀
# ============================================================
TICKERS = [
    'KO', 'PEP', 'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX', 'V', 'MA', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T', 'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW', 'WMT', 'TGT',
]

print('=' * 70)
print('1. 데이터 + 페어 발굴')
print('=' * 70)
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)
pairs = finder.screen_pairs(data)

# GDX~KO 강제 추가 (핵심 비교 대상)
train_cut = int(len(data) * 0.7)
train_data = data.iloc[:train_cut]
gdx_ko_beta = finder.tls_beta(
    np.log(train_data['KO']).values,
    np.log(train_data['GDX']).values)
spread_train = (np.log(train_data['GDX']) - gdx_ko_beta * np.log(train_data['KO']))
gdx_ko_hl = finder.half_life(spread_train) or 14.7
gdx_ko_pair = Pair(y='GDX', x='KO', beta=gdx_ko_beta,
                   pvalue=0.05, half_life=gdx_ko_hl)
pairs_to_test = pairs + [gdx_ko_pair]

print(f'  비교 대상: {len(pairs_to_test)}개 페어')


# ============================================================
# 2. 정적 vs 동적 K-fold 비교
# ============================================================
print()
print('=' * 70)
print('2. 정적 vs 동적 K-fold 비교 실행')
print('=' * 70)

kfold = PurgedKFold(n_splits=5, purge_days=30, embargo_pct=0.01)

results_compare = []
for p in pairs_to_test:
    sig_window = max(20, min(60, int(p.half_life * 2)))

    # 정적 베타
    static_report = kfold_evaluate_pair(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window, entry=2.0, exit_thr=0.5, stop=3.5,
    )

    # 동적 베타 (Kalman)
    dynamic_report = kfold_evaluate_pair_dynamic(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window, entry=2.0, exit_thr=0.5, stop=3.5,
        kalman_delta=1e-5, kalman_R=1e-3, kalman_warmup=60,
    )

    results_compare.append({
        'pair': p, 'static': static_report, 'dynamic': dynamic_report,
    })


# ============================================================
# 3. 종합 비교표
# ============================================================
print()
print('=' * 70)
print('3. 종합 비교: 정적 vs 동적')
print('=' * 70)
print()
print(f'  {"Pair":<14s} '
      f'{"정적 μ_R":>10s} {"동적 μ_R":>10s} {"ΔμR":>8s} '
      f'{"정적 SR":>9s} {"동적 SR":>9s} {"ΔSR":>8s} '
      f'{"정적 +F":>8s} {"동적 +F":>8s}')

improved_count = 0
for r in results_compare:
    s = r['static']
    d = r['dynamic']
    if s.total_folds == 0 or d.total_folds == 0:
        continue

    d_return = d.mean_val_return - s.mean_val_return
    d_sharpe = d.mean_val_sharpe - s.mean_val_sharpe
    is_improved = d_return > 0 and d_sharpe > 0
    if is_improved:
        improved_count += 1
    marker = '✓' if is_improved else ' '

    print(f'  {r["pair"].y+"~"+r["pair"].x:<14s} '
          f'{s.mean_val_return:>+10.2%} '
          f'{d.mean_val_return:>+10.2%} '
          f'{d_return:>+8.2%} '
          f'{s.mean_val_sharpe:>+9.2f} '
          f'{d.mean_val_sharpe:>+9.2f} '
          f'{d_sharpe:>+8.2f} '
          f'{s.profitable_folds:>3d}/{s.total_folds:<3d} '
          f'{d.profitable_folds:>3d}/{d.total_folds:<3d} {marker}')

print(f'\n  개선된 페어: {improved_count}/{len(results_compare)}')


# ============================================================
# 4. GDX~KO 상세 분석 (핵심 가설 검증)
# ============================================================
print()
print('=' * 70)
print('4. GDX~KO 상세 — 정적 vs 동적 fold별 비교')
print('=' * 70)

gdx_ko_result = next((r for r in results_compare
                      if r['pair'].y == 'GDX' and r['pair'].x == 'KO'), None)

if gdx_ko_result:
    s = gdx_ko_result['static']
    d = gdx_ko_result['dynamic']
    print()
    print('  Fold별 비교 (Val Sharpe, Val Return, Kelly):')
    print(f'  {"Fold":<6s} '
          f'{"정적 SR":>10s} {"동적 SR":>10s} {"ΔSR":>8s}   '
          f'{"정적 R":>9s} {"동적 R":>9s} {"ΔR":>8s}   '
          f'{"정적 K":>8s} {"동적 K":>8s}')
    for sf, df in zip(s.fold_results, d.fold_results):
        d_sr = df.val_sharpe - sf.val_sharpe
        d_r = df.val_total_return - sf.val_total_return
        marker = ' ✓' if (d_sr > 0 and d_r > 0) else (' ✗' if d_sr < 0 else '')
        print(f'  {sf.fold_idx:<6d} '
              f'{sf.val_sharpe:>+10.2f} '
              f'{df.val_sharpe:>+10.2f} '
              f'{d_sr:>+8.2f}   '
              f'{sf.val_total_return:>+9.2%} '
              f'{df.val_total_return:>+9.2%} '
              f'{d_r:>+8.2%}   '
              f'{sf.kelly_f:>8.2%} '
              f'{df.kelly_f:>8.2%}{marker}')

    print()
    print(f'  Fold 3 (1차 MVP에서 -0.29 Sharpe로 망했던 구간):')
    fold3_s = s.fold_results[3] if len(s.fold_results) > 3 else None
    fold3_d = d.fold_results[3] if len(d.fold_results) > 3 else None
    if fold3_s and fold3_d:
        print(f'    정적: Sharpe {fold3_s.val_sharpe:+.2f}, '
              f'Return {fold3_s.val_total_return:+.2%}')
        print(f'    동적: Sharpe {fold3_d.val_sharpe:+.2f}, '
              f'Return {fold3_d.val_total_return:+.2%}')
        delta = fold3_d.val_sharpe - fold3_s.val_sharpe
        print(f'    변화: ΔSharpe {delta:+.2f} '
              f'({"개선" if delta > 0 else "악화"})')


# ============================================================
# 5. 시각화
# ============================================================
print()
print('=' * 70)
print('5. 시각화')
print('=' * 70)

if gdx_ko_result:
    p = gdx_ko_result['pair']
    py = data[p.y]
    px = data[p.x]

    # 정적 spread
    static_spread = compute_spread(py, px, p.beta)
    # 동적 spread
    dynamic_beta, dynamic_spread = compute_dynamic_spread(
        py, px, delta=1e-5, R=1e-3, warmup=60)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))

    # (1,1) 베타 비교: 정적 평선 vs 동적 곡선
    ax = axes[0, 0]
    ax.axhline(p.beta, color='steelblue', lw=2, ls='-',
               label=f'Static beta = {p.beta:.3f}')
    ax.plot(dynamic_beta.index, dynamic_beta, color='darkorange', lw=1.5,
            label='Kalman dynamic beta')
    ax.axvline(data.index[train_cut], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Beta')
    ax.set_title(f'{p.y}~{p.x}: Beta trajectory')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (1,2) Spread 비교
    ax = axes[0, 1]
    ax.plot(static_spread.index, static_spread, color='steelblue', lw=1,
            alpha=0.7, label='Static spread')
    ax.plot(dynamic_spread.index, dynamic_spread, color='darkorange', lw=1,
            alpha=0.7, label='Dynamic spread')
    # Train mean lines
    train_idx = static_spread.index[:train_cut]
    ax.axhline(static_spread.loc[train_idx].mean(), color='steelblue',
               ls=':', alpha=0.5, label='Static train mean')
    ax.axhline(dynamic_spread.loc[train_idx].mean(), color='darkorange',
               ls=':', alpha=0.5, label='Dynamic train mean')
    ax.axvline(data.index[train_cut], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Spread')
    ax.set_title('Spread comparison (regime stability)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # (2,1) Fold별 Sharpe boxplot
    ax = axes[1, 0]
    static_sharpes = [[f.val_sharpe for f in r['static'].fold_results]
                      for r in results_compare
                      if r['static'].total_folds > 0]
    dynamic_sharpes = [[f.val_sharpe for f in r['dynamic'].fold_results]
                       for r in results_compare
                       if r['dynamic'].total_folds > 0]
    pair_labels = [f'{r["pair"].y}~{r["pair"].x}' for r in results_compare
                   if r['static'].total_folds > 0]
    positions_s = np.arange(len(pair_labels)) * 2.5
    positions_d = positions_s + 1
    bp1 = ax.boxplot(static_sharpes, positions=positions_s, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='steelblue'))
    bp2 = ax.boxplot(dynamic_sharpes, positions=positions_d, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='peachpuff'),
                     medianprops=dict(color='darkorange'))
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xticks(positions_s + 0.5)
    ax.set_xticklabels(pair_labels, rotation=30, ha='right')
    ax.set_ylabel('Val Sharpe per fold')
    ax.set_title('Fold Sharpe: Static (blue) vs Dynamic (orange)')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Static', 'Dynamic'],
              loc='upper right')
    ax.grid(alpha=0.3)

    # (2,2) Fold별 Return boxplot
    ax = axes[1, 1]
    static_returns = [[f.val_total_return for f in r['static'].fold_results]
                      for r in results_compare
                      if r['static'].total_folds > 0]
    dynamic_returns = [[f.val_total_return for f in r['dynamic'].fold_results]
                       for r in results_compare
                       if r['dynamic'].total_folds > 0]
    bp1 = ax.boxplot(static_returns, positions=positions_s, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
    bp2 = ax.boxplot(dynamic_returns, positions=positions_d, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='peachpuff'))
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xticks(positions_s + 0.5)
    ax.set_xticklabels(pair_labels, rotation=30, ha='right')
    ax.set_ylabel('Val Return per fold')
    ax.set_title('Fold Return: Static (blue) vs Dynamic (orange)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.grid(alpha=0.3)

    # (3,1) Mean Sharpe scatter (with error bars)
    ax = axes[2, 0]
    for r in results_compare:
        if r['static'].total_folds == 0:
            continue
        s = r['static']
        d = r['dynamic']
        ax.errorbar(s.mean_val_sharpe, d.mean_val_sharpe,
                    xerr=s.std_val_sharpe, yerr=d.std_val_sharpe,
                    fmt='o', markersize=10, capsize=4, alpha=0.7,
                    label=f'{r["pair"].y}~{r["pair"].x}')
    lims = [-2, 8]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x (no change)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Static mean Val Sharpe')
    ax.set_ylabel('Dynamic mean Val Sharpe')
    ax.set_title('Static vs Dynamic — points above y=x mean improvement')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    # (3,2) GDX~KO Fold별 Sharpe 비교 막대
    ax = axes[2, 1]
    if gdx_ko_result:
        s = gdx_ko_result['static']
        d = gdx_ko_result['dynamic']
        n_folds = len(s.fold_results)
        x = np.arange(n_folds)
        width = 0.35
        s_sharpes = [f.val_sharpe for f in s.fold_results]
        d_sharpes = [f.val_sharpe for f in d.fold_results]
        bars1 = ax.bar(x - width/2, s_sharpes, width, color='steelblue',
                       alpha=0.7, label='Static')
        bars2 = ax.bar(x + width/2, d_sharpes, width, color='darkorange',
                       alpha=0.7, label='Dynamic')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
        ax.set_ylabel('Val Sharpe')
        ax.set_title('GDX~KO: Fold-by-fold Sharpe (Static vs Dynamic)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = '/home/claude/kalman_comparison_chart.png'
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {out_path}')


# ============================================================
# 6. 가설 검증 결과
# ============================================================
print()
print('=' * 70)
print('6. 핵심 가설 검증')
print('=' * 70)
print()
print('  가설: "Kalman 동적 베타가 GDX~KO의 fold 3 손실을 줄여준다."')
if gdx_ko_result:
    fold3_s = gdx_ko_result['static'].fold_results[3] if \
              len(gdx_ko_result['static'].fold_results) > 3 else None
    fold3_d = gdx_ko_result['dynamic'].fold_results[3] if \
              len(gdx_ko_result['dynamic'].fold_results) > 3 else None
    if fold3_s and fold3_d:
        if fold3_d.val_sharpe > fold3_s.val_sharpe:
            print(f'  결과: ✓ 가설 지지')
            print(f'    Fold 3 Sharpe: {fold3_s.val_sharpe:+.2f} → '
                  f'{fold3_d.val_sharpe:+.2f}')
            print(f'    → Kalman을 표준 도구로 채택, 다른 페어에도 적용 권장.')
        else:
            print(f'  결과: ✗ 가설 기각')
            print(f'    Fold 3 Sharpe: {fold3_s.val_sharpe:+.2f} → '
                  f'{fold3_d.val_sharpe:+.2f}')
            print(f'    → Kalman만으로는 부족. 롤링 페어 재발굴이 정답.')

print()
print(f'  전체 개선도: {improved_count}/{len(results_compare)} 페어가 동적 베타로 개선됨')
