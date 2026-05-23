"""
5차 MVP: 정적 vs 동적 vs Ensemble 3-way 비교
=============================================

가설:
  "최근 성과에 따라 정적·동적 베타에 자동으로 자본을 배분하는 Ensemble은
   각 시장 국면의 강한 쪽이 우세해져, 평균적으로 두 전략 모두를 능가한다."

검증:
  1. 5개 페어에 대해 3개 전략 K-fold 평가
  2. 평균 Sharpe / 평균 Return / 견고성 점수 비교
  3. GDX~KO fold별 비교 (각 fold에서 누가 최강인지)
  4. Ensemble의 가중치 추이 시각화 (시장 국면에 어떻게 반응하는지)
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
    kfold_evaluate_pair_dynamic,
    compute_dynamic_spread,
)
from ensemble_strategy import (
    EnsembleBacktester,
    kfold_evaluate_ensemble,
    adaptive_weight,
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
print('1. 데이터 + 페어 풀')
print('=' * 70)
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)
pairs = finder.screen_pairs(data)

# GDX~KO 강제 추가
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
# 2. 3-way K-fold 평가
# ============================================================
print()
print('=' * 70)
print('2. 정적 vs 동적 vs Ensemble — K-fold 비교')
print('=' * 70)

kfold = PurgedKFold(n_splits=5, purge_days=30, embargo_pct=0.01)

three_way = []
for p in pairs_to_test:
    sig_window = max(20, min(60, int(p.half_life * 2)))

    static_r = kfold_evaluate_pair(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window, entry=2.0, exit_thr=0.5, stop=3.5)

    dynamic_r = kfold_evaluate_pair_dynamic(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window, entry=2.0, exit_thr=0.5, stop=3.5)

    ensemble_r = kfold_evaluate_ensemble(
        data[p.y], data[p.x], p, kfold,
        sig_window=sig_window, lookback=30, temperature=5.0,
        min_weight=0.1, kalman_delta=1e-5)

    three_way.append({
        'pair': p,
        'static': static_r,
        'dynamic': dynamic_r,
        'ensemble': ensemble_r,
    })


# ============================================================
# 3. 종합 비교표
# ============================================================
print()
print('=' * 70)
print('3. 3-way 비교 — Sharpe 평균')
print('=' * 70)
print()
print(f'  {"페어":<14s} '
      f'{"정적 SR":>10s} {"동적 SR":>10s} {"앙상블 SR":>11s}   '
      f'{"최강":>8s}')

best_counts = {'정적': 0, '동적': 0, '앙상블': 0}
for r in three_way:
    s = r['static']
    d = r['dynamic']
    e = r['ensemble']
    if s.total_folds == 0 or d.total_folds == 0 or e.total_folds == 0:
        continue

    sr = {'정적': s.mean_val_sharpe,
          '동적': d.mean_val_sharpe,
          '앙상블': e.mean_val_sharpe}
    winner = max(sr, key=sr.get)
    best_counts[winner] += 1

    print(f'  {r["pair"].y+"~"+r["pair"].x:<14s} '
          f'{s.mean_val_sharpe:>+10.2f} '
          f'{d.mean_val_sharpe:>+10.2f} '
          f'{e.mean_val_sharpe:>+11.2f}   '
          f'{winner:>8s}')

print(f'\n  최강 횟수: {best_counts}')


# ============================================================
# 4. GDX~KO Fold별 상세
# ============================================================
print()
print('=' * 70)
print('4. GDX~KO Fold별 — 누가 어디서 강한가')
print('=' * 70)
gdx_ko = next((r for r in three_way
               if r['pair'].y == 'GDX' and r['pair'].x == 'KO'), None)
if gdx_ko:
    s = gdx_ko['static']
    d = gdx_ko['dynamic']
    e = gdx_ko['ensemble']
    print()
    print(f'  {"Fold":<6s} '
          f'{"정적 SR":>10s} {"동적 SR":>10s} {"앙상블 SR":>11s}   {"승자":>8s}')
    for sf, df, ef in zip(s.fold_results, d.fold_results, e.fold_results):
        sr = {'정적': sf.val_sharpe,
              '동적': df.val_sharpe,
              '앙상블': ef.val_sharpe}
        winner = max(sr, key=sr.get)
        print(f'  {sf.fold_idx:<6d} '
              f'{sf.val_sharpe:>+10.2f} '
              f'{df.val_sharpe:>+10.2f} '
              f'{ef.val_sharpe:>+11.2f}   {winner:>8s}')


# ============================================================
# 5. 시각화
# ============================================================
print()
print('=' * 70)
print('5. 시각화')
print('=' * 70)

if gdx_ko:
    p = gdx_ko['pair']
    py = data[p.y]
    px = data[p.x]

    # 전체 Ensemble 백테스트 (가중치 추이 시각화용)
    bt = EnsembleBacktester(initial_capital=100_000,
                            capital_fraction=1.0,
                            lookback=30, temperature=5.0, min_weight=0.1)
    ens_full = bt.run(py, px, p,
                      sig_window=max(20, min(60, int(p.half_life * 2))),
                      kalman_delta=1e-5)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))

    # (1,1) 세 전략의 Equity 비교 (가상, 같은 자본)
    ax = axes[0, 0]
    # 정적
    sig_gen_local = compute_spread(py, px, p.beta)
    from pairs_trading_mvp import SignalGenerator, Backtester
    sg = SignalGenerator(window=max(20, min(60, int(p.half_life * 2))),
                         entry=2.0, exit_thr=0.5, stop=3.5)
    z_s, pos_s, fc_s = sg.generate(sig_gen_local)
    bt_s = Backtester(initial_capital=100_000, capital_fraction=1.0)
    res_s = bt_s.run(py, px, p.beta, pos_s, fc_s)

    # 동적
    beta_d, spread_d = compute_dynamic_spread(py, px, delta=1e-5, R=1e-3, warmup=60)
    z_d, pos_d, fc_d = sg.generate(spread_d)
    from kalman_pairs import DynamicBetaBacktester
    bt_d = DynamicBetaBacktester(initial_capital=100_000, capital_fraction=1.0)
    res_d = bt_d.run(py, px, beta_d, pos_d, fc_d)

    ax.plot(res_s.equity.index, res_s.equity, color='steelblue',
            lw=1.5, label='Static beta')
    ax.plot(res_d.equity.index, res_d.equity, color='darkorange',
            lw=1.5, label='Kalman dynamic beta')
    ax.plot(ens_full.equity.index, ens_full.equity, color='darkgreen',
            lw=2, label='Ensemble (adaptive)')
    ax.axvline(data.index[train_cut], color='red', ls='--', alpha=0.5,
               label='Train/Test split')
    ax.set_ylabel('Equity (USD)')
    ax.set_title(f'{p.y}~{p.x}: Equity curves — three strategies')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (1,2) Ensemble 가중치 추이
    ax = axes[0, 1]
    ax.plot(ens_full.weights_static.index, ens_full.weights_static,
            color='steelblue', lw=1.5, label='Static weight')
    ax.plot(ens_full.weights_dynamic.index, ens_full.weights_dynamic,
            color='darkorange', lw=1.5, label='Dynamic weight', alpha=0.7)
    ax.axhline(0.5, color='black', lw=0.5, alpha=0.5)
    ax.axvline(data.index[train_cut], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Capital weight')
    ax.set_title('Ensemble auto-allocation (responds to market regime)')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # (2,1) 페어별 Sharpe 막대 비교
    ax = axes[1, 0]
    valid = [r for r in three_way if r['static'].total_folds > 0]
    n_pairs = len(valid)
    pair_labels = [f'{r["pair"].y}~{r["pair"].x}' for r in valid]
    static_sr = [r['static'].mean_val_sharpe for r in valid]
    dynamic_sr = [r['dynamic'].mean_val_sharpe for r in valid]
    ensemble_sr = [r['ensemble'].mean_val_sharpe for r in valid]

    x = np.arange(n_pairs)
    w = 0.27
    ax.bar(x - w, static_sr, w, color='steelblue', alpha=0.8, label='Static')
    ax.bar(x, dynamic_sr, w, color='darkorange', alpha=0.8, label='Dynamic')
    ax.bar(x + w, ensemble_sr, w, color='darkgreen', alpha=0.8, label='Ensemble')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30, ha='right')
    ax.set_ylabel('Mean Val Sharpe')
    ax.set_title('Per-pair Mean Sharpe — 3-way comparison')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (2,2) GDX~KO fold별 Sharpe 비교 막대
    ax = axes[1, 1]
    s = gdx_ko['static']
    d = gdx_ko['dynamic']
    e = gdx_ko['ensemble']
    n_folds = len(s.fold_results)
    x = np.arange(n_folds)
    s_sr = [f.val_sharpe for f in s.fold_results]
    d_sr = [f.val_sharpe for f in d.fold_results]
    e_sr = [f.val_sharpe for f in e.fold_results]
    ax.bar(x - w, s_sr, w, color='steelblue', alpha=0.8, label='Static')
    ax.bar(x, d_sr, w, color='darkorange', alpha=0.8, label='Dynamic')
    ax.bar(x + w, e_sr, w, color='darkgreen', alpha=0.8, label='Ensemble')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.set_ylabel('Val Sharpe')
    ax.set_title('GDX~KO: Fold-by-fold Sharpe — 3 strategies')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (3,1) 가중치 vs 시장 국면
    ax = axes[2, 0]
    avg_w_s = ens_full.weights_static.rolling(20).mean()
    ax.fill_between(avg_w_s.index, 0.5, avg_w_s,
                    where=(avg_w_s > 0.5),
                    color='steelblue', alpha=0.3, label='Static-favored regime')
    ax.fill_between(avg_w_s.index, avg_w_s, 0.5,
                    where=(avg_w_s <= 0.5),
                    color='darkorange', alpha=0.3, label='Dynamic-favored regime')
    ax.plot(avg_w_s.index, avg_w_s, color='black', lw=1, alpha=0.7)
    ax.axhline(0.5, color='black', lw=0.5, ls='--')
    ax.axvline(data.index[train_cut], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Static weight (20d MA)')
    ax.set_title('Ensemble regime detection — who did it trust more?')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # (3,2) Total Return 종합 비교 — Drawdown으로 변경 (더 정직)
    ax = axes[2, 1]
    # 각 페어의 Mean OOS Return 또는 max_drawdown 표시
    static_mdd = [abs(min([f.val_max_drawdown for f in r['static'].fold_results])) for r in valid]
    dynamic_mdd = [abs(min([f.val_max_drawdown for f in r['dynamic'].fold_results])) for r in valid]
    ensemble_mdd = [abs(min([f.val_max_drawdown for f in r['ensemble'].fold_results])) for r in valid]
    ax.bar(x - w, static_mdd, w, color='steelblue', alpha=0.8, label='Static')
    ax.bar(x, dynamic_mdd, w, color='darkorange', alpha=0.8, label='Dynamic')
    ax.bar(x + w, ensemble_mdd, w, color='darkgreen', alpha=0.8, label='Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30, ha='right')
    ax.set_ylabel('Worst fold max drawdown (lower = better)')
    ax.set_title('Worst-case Drawdown per pair — does ensemble stabilize?')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = '/home/claude/ensemble_comparison_chart.png'
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {out_path}')


# ============================================================
# 6. 결론
# ============================================================
print()
print('=' * 70)
print('6. 핵심 결론')
print('=' * 70)
print()
print(f'  최강 카운트: {best_counts}')
print(f'  → 앙상블이 {best_counts["앙상블"]}/{sum(best_counts.values())} 페어에서 최강')

if gdx_ko:
    e = gdx_ko['ensemble']
    s = gdx_ko['static']
    d = gdx_ko['dynamic']
    print()
    print('  GDX~KO 종합:')
    print(f'    정적 평균 Sharpe : {s.mean_val_sharpe:+.2f}')
    print(f'    동적 평균 Sharpe : {d.mean_val_sharpe:+.2f}')
    print(f'    앙상블 평균 Sharpe: {e.mean_val_sharpe:+.2f}')

    avg_w_s = ens_full.weights_static.mean()
    print(f'    앙상블의 평균 정적 비중: {avg_w_s:.1%}')
    print(f'    (50%면 균형, >50%면 평소 정적 의존, <50%면 평소 동적 의존)')
