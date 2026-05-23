"""
9차 MVP-D: Refresh Frequency Sweep
======================================

9차-B 발견 ① 직접 검증:
  "refresh_every_days=30이 cooldown granularity의 자연 단위"
  "cd ≤ refresh_every인 cooldown은 모두 동일 효과"

이게 사실이라면:
  - refresh=60일 때 cd=30 cooldown은 무의미 (refresh가 자연 단위니까)
  - refresh가 늘어날수록 cd=0과 cd=30의 차이가 사라져야 함

추가 검증:
  "refresh 자체를 늘리면 (롤링 회전율 감소) 손실 축소된다"

  9차-A에서 본 -3.50 bp/일 음수 PnL 구간은 신규 페어 진입 직후.
  refresh를 늘리면 신규 진입 빈도가 줄어듦.
  → cooldown 없이도 자연스럽게 갭 축소되어야

2D Grid sweep:
  refresh ∈ {30, 60, 90, 180}일  × cooldown ∈ {0, 30}일 = 8 변형

기대되는 결과 시나리오:
  α. refresh↑ → cd=0과 cd=30 수렴 (발견 ① 입증)
  β. refresh↑ → cd=0 자체가 (b)에 근접 (회전율 자체가 문제)
  γ. refresh↑해도 (b)에 도달 못 함 (페어 풀 품질 문제 → 9-C로)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, Backtester, compute_spread,
)
from production_system import ProductionSystem


# ============================================================
# 1. 데이터
# ============================================================
TICKERS = [
    'KO', 'PEP', 'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX', 'V', 'MA', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T', 'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW', 'WMT', 'TGT',
]

print('=' * 72)
print('9차 MVP-D: Refresh × Cooldown 2D sweep')
print('=' * 72)
print()
print('1. 데이터 다운로드')
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'   {data.shape[0]}일 × {data.shape[1]}종목')

INITIAL_CAPITAL = 100_000

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)


# ============================================================
# 2. (b) 재현 — 참조 기준선
# ============================================================
print()
print('2. (b) 고정 풀 재현 (참조)')
train_split = int(len(data) * 0.7)
fixed_pool = finder.screen_pairs(data.iloc[:train_split])
daily_returns_b = pd.Series(0.0, index=data.index)
equal_w = 0.10 / len(fixed_pool)
for p in fixed_pool:
    spread = compute_spread(data[p.y], data[p.x], p.beta)
    sw = max(20, min(60, int(p.half_life * 2)))
    sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
    z, pos, fc = sg.generate(spread)
    bt = Backtester(initial_capital=INITIAL_CAPITAL,
                    capital_fraction=1.0,
                    fee_rate=0.0004, slippage=0.0005)
    res = bt.run(data[p.y], data[p.x], p.beta, pos, fc)
    daily_returns_b += equal_w * res.equity.pct_change().fillna(0)
equity_b = INITIAL_CAPITAL * (1 + daily_returns_b).cumprod()
ret_b = equity_b.iloc[-1] / INITIAL_CAPITAL - 1
sh_b_series = equity_b.pct_change().dropna()
sharpe_b = sh_b_series.mean() / sh_b_series.std() * np.sqrt(252) if sh_b_series.std() > 0 else 0
mdd_b = float((equity_b / equity_b.cummax() - 1).min())
print(f'   (b) 최종 ${equity_b.iloc[-1]:,.0f} ({ret_b:+.2%}), Sharpe {sharpe_b:+.2f}')


# ============================================================
# 3. 2D Grid Sweep
# ============================================================
print()
print('=' * 72)
print('3. 2D Grid sweep — refresh × cooldown')
print('=' * 72)

refresh_values = [30, 60, 90, 180]
cooldown_values = [0, 30]

common_kwargs = dict(
    finder=finder,
    lookback_days=365,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=INITIAL_CAPITAL,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=True,
    max_active_pairs=5,
    min_historical_sharpe=0.0,
    quality_lookback=90,
)

# 결과 저장 (refresh, cooldown) → metrics
results = {}

for refresh in refresh_values:
    for cd in cooldown_values:
        print()
        print(f'  refresh={refresh:>3d}d  cooldown={cd:>2d}d ...', end=' ', flush=True)
        sys_v = ProductionSystem(
            **common_kwargs,
            refresh_every_days=refresh,
            pair_cooldown_days=cd,
        )
        res = sys_v.run(data, verbose=False)
        eq = res.equity_curve
        m = res.metrics

        # 추가 통계
        n_snapshots = len(res.monthly_states)
        avg_active = np.mean([len(s.active_pairs) for s in res.monthly_states
                              if s.active_pairs]) if any(s.active_pairs for s in res.monthly_states) else 0

        # 페어 회전 횟수
        prev_pids = set()
        n_new_total = 0
        for st in res.monthly_states:
            cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
            n_new_total += len(cur_pids - prev_pids)
            prev_pids = cur_pids

        # 고유 페어 수
        all_pair_ids = set()
        for st in res.monthly_states:
            for p in st.active_pairs:
                all_pair_ids.add(f'{p.y}~{p.x}')

        results[(refresh, cd)] = {
            'equity': eq,
            'final': float(eq.iloc[-1]),
            'return': m['total_return'],
            'sharpe': m['sharpe'],
            'mdd': m['max_drawdown'],
            'n_snapshots': n_snapshots,
            'avg_active_pairs': avg_active,
            'n_new_total': n_new_total,
            'n_unique_pairs': len(all_pair_ids),
        }
        print(f'final ${eq.iloc[-1]:>9,.0f}  Ret {m["total_return"]:+7.2%}  '
              f'Sh {m["sharpe"]:+5.2f}  MDD {m["max_drawdown"]:+6.2%}  '
              f'pairs {avg_active:.1f}')


# ============================================================
# 4. 결과 테이블
# ============================================================
print()
print('=' * 72)
print('4. 종합 비교 테이블')
print('=' * 72)
print()
print(f'  {"Refresh":>8s} {"CD":>4s}  {"최종":>10s} {"Return":>8s} {"Sharpe":>7s} '
      f'{"MDD":>7s} {"snap":>5s} {"pairs":>6s} {"new":>5s} {"unique":>7s}')
print(f'  {"-"*8} {"-"*4}  {"-"*10} {"-"*8} {"-"*7} {"-"*7} {"-"*5} {"-"*6} {"-"*5} {"-"*7}')

for refresh in refresh_values:
    for cd in cooldown_values:
        r = results[(refresh, cd)]
        print(f'  {refresh:>6d}d  {cd:>2d}d  '
              f'${r["final"]:>8,.0f}  '
              f'{r["return"]:>+7.2%} '
              f'{r["sharpe"]:>+7.2f} '
              f'{r["mdd"]:>+6.2%} '
              f'{r["n_snapshots"]:>5d} '
              f'{r["avg_active_pairs"]:>6.1f} '
              f'{r["n_new_total"]:>5d} '
              f'{r["n_unique_pairs"]:>7d}')
    print()
# 참조
print(f'  {"(b)":>8s} {"-":>4s}  '
      f'${equity_b.iloc[-1]:>8,.0f}  '
      f'{ret_b:>+7.2%} '
      f'{sharpe_b:>+7.2f} '
      f'{mdd_b:>+6.2%}')


# ============================================================
# 5. 발견 ① 검증 — cd=0과 cd=30의 갭이 refresh↑로 줄어드는가
# ============================================================
print()
print('=' * 72)
print('5. 발견 ① 검증: refresh↑ 시 cooldown 효과 사라지는가')
print('=' * 72)
print()
print(f'  {"Refresh":>8s} {"cd=0 Ret":>10s} {"cd=30 Ret":>11s} {"Δ":>8s} '
      f'{"cd=0 Sh":>9s} {"cd=30 Sh":>10s} {"Δ":>6s}')
print(f'  {"-"*8} {"-"*10} {"-"*11} {"-"*8} {"-"*9} {"-"*10} {"-"*6}')

cd_effects_return = []
cd_effects_sharpe = []
for refresh in refresh_values:
    r0 = results[(refresh, 0)]
    r30 = results[(refresh, 30)]
    delta_ret = r30['return'] - r0['return']
    delta_sh = r30['sharpe'] - r0['sharpe']
    cd_effects_return.append(delta_ret)
    cd_effects_sharpe.append(delta_sh)
    print(f'  {refresh:>6d}d  '
          f'{r0["return"]:>+9.2%}  '
          f'{r30["return"]:>+10.2%}  '
          f'{delta_ret:>+7.2%}  '
          f'{r0["sharpe"]:>+8.2f}  '
          f'{r30["sharpe"]:>+9.2f}  '
          f'{delta_sh:>+5.2f}')

print()
# 단조 감소 검증 (cooldown 효과가 refresh↑로 줄어드는지)
abs_effects = [abs(d) for d in cd_effects_return]
monotone_decline = all(abs_effects[i] >= abs_effects[i + 1]
                       for i in range(len(abs_effects) - 1))
print(f'  cooldown 효과 |Δ| (refresh 30→180): {[f"{x*100:+.2f}%p" for x in cd_effects_return]}')
print(f'  단조 감소? {monotone_decline}')


# ============================================================
# 6. 시나리오 판정
# ============================================================
print()
print('=' * 72)
print('6. 시나리오 판정')
print('=' * 72)
print()

# (b) 도달 여부
beat_b = []
for refresh in refresh_values:
    for cd in cooldown_values:
        r = results[(refresh, cd)]
        if r['return'] >= ret_b:
            beat_b.append((refresh, cd, r['return']))

if beat_b:
    print(f'  ★ (b) 도달: {beat_b}')
else:
    closest = max(((refresh, cd, results[(refresh, cd)]['return'])
                  for refresh in refresh_values for cd in cooldown_values),
                  key=lambda x: x[2])
    gap = ret_b - closest[2]
    print(f'  ✗ 어떤 조합도 (b) +{ret_b*100:.2f}%에 도달 못 함')
    print(f'    가장 근접: refresh={closest[0]}d, cd={closest[1]}d → {closest[2]*100:+.2f}% '
          f'(잔여 갭 {gap*100:+.2f}%p)')

# 최고 변형
best = max(results.items(), key=lambda x: x[1]['sharpe'])
print(f'  최고 Sharpe: refresh={best[0][0]}d, cd={best[0][1]}d → Sharpe {best[1]["sharpe"]:+.2f}')


# ============================================================
# 7. 시각화
# ============================================================
print()
print('=' * 72)
print('7. 시각화')
print('=' * 72)

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full row) 모든 8개 변형 + (b) equity curves
ax = fig.add_subplot(gs[0, :])
colors_refresh = {30: 'darkblue', 60: 'teal', 90: 'darkgreen', 180: 'darkorange'}
for (refresh, cd), r in results.items():
    color = colors_refresh[refresh]
    lw = 1.8 if cd == 0 else 1.4
    ls = '-' if cd == 0 else '--'
    label = f'refresh={refresh}d, cd={cd}d → {r["return"]:+.2%}'
    ax.plot(r['equity'].index, r['equity'], label=label,
            color=color, lw=lw, ls=ls, alpha=0.85)
ax.plot(equity_b.index, equity_b,
        label=f'(b) Fixed pool ref → {ret_b:+.2%}',
        color='red', lw=2.2, ls=':')
ax.axhline(INITIAL_CAPITAL, color='black', ls=':', alpha=0.4)
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title('MVP 9-D: Refresh × Cooldown grid + (b) reference')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) Heatmap: Return
ax = fig.add_subplot(gs[1, 0])
heatmap_ret = np.zeros((len(cooldown_values), len(refresh_values)))
for i, cd in enumerate(cooldown_values):
    for j, refresh in enumerate(refresh_values):
        heatmap_ret[i, j] = results[(refresh, cd)]['return'] * 100

im = ax.imshow(heatmap_ret, cmap='RdYlGn', aspect='auto',
               vmin=-10, vmax=5)
ax.set_xticks(range(len(refresh_values)))
ax.set_xticklabels([f'{r}d' for r in refresh_values])
ax.set_yticks(range(len(cooldown_values)))
ax.set_yticklabels([f'cd={c}d' for c in cooldown_values])
ax.set_xlabel('refresh_every_days')
ax.set_title(f'Total Return (%) — (b) ref = {ret_b*100:+.2f}%')

for i in range(len(cooldown_values)):
    for j in range(len(refresh_values)):
        v = heatmap_ret[i, j]
        color = 'black' if abs(v) < 5 else 'white'
        ax.text(j, i, f'{v:+.2f}%', ha='center', va='center',
                color=color, fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046)

# (2, 2) Heatmap: Sharpe
ax = fig.add_subplot(gs[1, 1])
heatmap_sh = np.zeros((len(cooldown_values), len(refresh_values)))
for i, cd in enumerate(cooldown_values):
    for j, refresh in enumerate(refresh_values):
        heatmap_sh[i, j] = results[(refresh, cd)]['sharpe']

im2 = ax.imshow(heatmap_sh, cmap='RdYlGn', aspect='auto',
                vmin=-1.5, vmax=1.0)
ax.set_xticks(range(len(refresh_values)))
ax.set_xticklabels([f'{r}d' for r in refresh_values])
ax.set_yticks(range(len(cooldown_values)))
ax.set_yticklabels([f'cd={c}d' for c in cooldown_values])
ax.set_xlabel('refresh_every_days')
ax.set_title(f'Sharpe ratio — (b) ref = {sharpe_b:+.2f}')

for i in range(len(cooldown_values)):
    for j in range(len(refresh_values)):
        v = heatmap_sh[i, j]
        color = 'black' if abs(v) < 0.5 else 'white'
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                color=color, fontweight='bold', fontsize=11)
plt.colorbar(im2, ax=ax, fraction=0.046)

# (3, 1) 발견 ① 검증: cd=0과 cd=30의 차이
ax = fig.add_subplot(gs[2, 0])
x = np.arange(len(refresh_values))
returns_cd0 = [results[(r, 0)]['return'] * 100 for r in refresh_values]
returns_cd30 = [results[(r, 30)]['return'] * 100 for r in refresh_values]
ax.plot(x, returns_cd0, 'o-', color='steelblue', lw=2.5, markersize=10,
        label='cooldown=0d')
ax.plot(x, returns_cd30, 's-', color='darkorange', lw=2.5, markersize=10,
        label='cooldown=30d')
ax.fill_between(x, returns_cd0, returns_cd30, alpha=0.2, color='gray',
                label='cooldown effect')
ax.axhline(ret_b * 100, color='red', ls=':', alpha=0.7,
           label=f'(b) ref {ret_b*100:+.2f}%')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'{r}d' for r in refresh_values])
ax.set_xlabel('refresh_every_days')
ax.set_ylabel('Total Return (%)')
ax.set_title('Discovery ① test — does cooldown effect shrink as refresh grows?')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (3, 2) 페어 회전 횟수 vs refresh
ax = fig.add_subplot(gs[2, 1])
new_counts_cd0 = [results[(r, 0)]['n_new_total'] for r in refresh_values]
unique_counts_cd0 = [results[(r, 0)]['n_unique_pairs'] for r in refresh_values]
new_counts_cd30 = [results[(r, 30)]['n_new_total'] for r in refresh_values]

w = 0.35
ax.bar(x - w / 2, new_counts_cd0, w, color='steelblue', alpha=0.85,
       label='# new entries (cd=0)')
ax.bar(x + w / 2, new_counts_cd30, w, color='darkorange', alpha=0.85,
       label='# new entries (cd=30)')
ax.set_xticks(x)
ax.set_xticklabels([f'{r}d' for r in refresh_values])
ax.set_xlabel('refresh_every_days')
ax.set_ylabel('Number of new pair entries')
ax.set_title('Pair turnover decreases with refresh')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('MVP 9-D: Refresh frequency × Cooldown interaction',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9d_refresh_sweep.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'   차트 저장: {out_path}')


# ============================================================
# 8. 결론 + 데이터 dump
# ============================================================
print()
print('=' * 72)
print('8. 결론')
print('=' * 72)
print()

# 최고 변형
best_key = max(results, key=lambda k: results[k]['sharpe'])
best_r = results[best_key]
print(f'  최고 Sharpe: refresh={best_key[0]}d, cd={best_key[1]}d')
print(f'    Return {best_r["return"]:+.2%}, Sharpe {best_r["sharpe"]:+.2f}, '
      f'MDD {best_r["mdd"]:+.2%}')

# refresh=180, cd=0 vs (b)
r180_cd0 = results[(180, 0)]
print()
print(f'  refresh=180일 + cd=0 (가장 (b)에 가까운 롤링) 평가:')
print(f'    Return  {r180_cd0["return"]:+.2%}  (vs (b) {ret_b:+.2%}, 갭 {(r180_cd0["return"]-ret_b)*100:+.2f}%p)')
print(f'    Sharpe  {r180_cd0["sharpe"]:+.2f}  (vs (b) {sharpe_b:+.2f}, 갭 {r180_cd0["sharpe"]-sharpe_b:+.2f})')

# 발견 ① 결과
print()
print(f'  발견 ① 검증 — cooldown 효과 |Δ| 변화 (refresh 30→180):')
for refresh, delta in zip(refresh_values, cd_effects_return):
    print(f'    refresh={refresh:>3d}d: |Δ|={abs(delta)*100:.2f}%p')
if abs(cd_effects_return[0]) > abs(cd_effects_return[-1]):
    print(f'  ✓ cooldown 효과가 refresh↑로 감소 — 발견 ① 지지')
else:
    print(f'  ✗ 비단조 — 발견 ① 부분 기각')

# 데이터 dump
import json
report_data = {
    'grid': {
        f'{refresh}_{cd}': {
            'refresh': refresh, 'cooldown': cd,
            'final': r['final'], 'return': r['return'],
            'sharpe': r['sharpe'], 'mdd': r['mdd'],
            'avg_pairs': r['avg_active_pairs'],
            'n_new': r['n_new_total'],
            'n_unique': r['n_unique_pairs'],
        }
        for (refresh, cd), r in results.items()
    },
    'reference_b': {
        'return': ret_b, 'sharpe': float(sharpe_b),
        'mdd': float(mdd_b),
    },
    'cooldown_effects': {
        f'refresh_{refresh}': cd_effects_return[i]
        for i, refresh in enumerate(refresh_values)
    },
    'discovery_1_supported': bool(abs(cd_effects_return[0]) > abs(cd_effects_return[-1])),
    'best_combo': {'refresh': best_key[0], 'cooldown': best_key[1],
                   'sharpe': best_r['sharpe']},
}
data_dump_path = '/tmp/mvp9d_report_data.json'
with open(data_dump_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print()
print(f'  데이터 저장: {data_dump_path}')
