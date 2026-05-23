"""
9차 MVP-B: 신규 페어 Cooldown 휴리스틱
========================================

9차-A 결과로부터:
  - (d3) 손실의 -9.78%p가 "롤링 효과"에서 옴 (총 -9.48%p 중)
  - 신규 페어 진입 후 30일 일별 평균 -3.50 bp (전체 평균 -0.85 bp의 4배 나쁨)
  - 진입 직후 양수 일수 비율 23.5% (정상은 ~50%)
  - 진입 직후 30일 누적 평균 -1.04%

가설:
  "신규 페어 진입 후 N일은 운용 보류(cooldown)하면, 그 음수 PnL 구간을
   회피하여 양수 수익으로 회복 가능하다."

Cooldown sweep:
  (d3-cd0)  baseline (cooldown=0)
  (d3-cd15) cooldown 15일
  (d3-cd30) cooldown 30일  ← 9차-A의 핵심 메트릭 윈도우와 일치
  (d3-cd60) cooldown 60일
  (d3-cd90) cooldown 90일  (공격적)

평가 차원:
  1. 수익률, Sharpe, MDD
  2. 운용 페어 수 / 시간 (cooldown 길수록 줄어들 것)
  3. 단조성: cooldown 늘릴수록 단조 개선되는가?
  4. (b) 고정 풀 +2.94%, Sharpe +0.83에 도달 가능한가?
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
print('9차 MVP-B: 신규 페어 Cooldown 휴리스틱 sweep')
print('=' * 72)
print()
print('1. 데이터 다운로드')
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'   {data.shape[0]}일 × {data.shape[1]}종목')

INITIAL_CAPITAL = 100_000


# ============================================================
# 2. (b) 재현 — 참조 기준선
# ============================================================
print()
print('2. (b) 고정 풀 재현 (참조 기준선)')
finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)
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
print(f'   (b) 최종 ${equity_b.iloc[-1]:,.0f} ({ret_b:+.2%})')


# ============================================================
# 3. Cooldown sweep (5 변형)
# ============================================================
print()
print('=' * 72)
print('3. Cooldown sweep — 5가지 cd 값')
print('=' * 72)

common_kwargs = dict(
    finder=finder,
    lookback_days=365,
    refresh_every_days=30,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=INITIAL_CAPITAL,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=True,
    max_active_pairs=5,
    min_historical_sharpe=0.0,
    quality_lookback=90,
)

cooldowns = [0, 15, 30, 60, 90]
variants = []
for cd in cooldowns:
    print()
    print(f'  cooldown = {cd:>2d}일')
    sys_v = ProductionSystem(**common_kwargs, pair_cooldown_days=cd)
    res = sys_v.run(data, verbose=False)
    eq = res.equity_curve
    m = res.metrics

    # cooldown으로 블락된 페어 수
    n_blocked_total = sum(len(e['blocked']) for e in sys_v.cooldown_log)
    n_blocked_unique = len(set(p for e in sys_v.cooldown_log for p in e['blocked']))

    # 평균 active 페어 수
    avg_active = np.mean([len(s.active_pairs) for s in res.monthly_states
                          if s.active_pairs]) if any(s.active_pairs for s in res.monthly_states) else 0

    variants.append({
        'cd': cd,
        'system': sys_v,
        'result': res,
        'equity': eq,
        'final': float(eq.iloc[-1]),
        'return': m['total_return'],
        'sharpe': m['sharpe'],
        'mdd': m['max_drawdown'],
        'avg_active_pairs': avg_active,
        'n_blocked_total': n_blocked_total,
        'n_blocked_unique': n_blocked_unique,
    })
    print(f'    → 최종 ${eq.iloc[-1]:>10,.0f}  '
          f'Return {m["total_return"]:+7.2%}  '
          f'Sharpe {m["sharpe"]:+5.2f}  '
          f'MDD {m["max_drawdown"]:+6.2%}')
    print(f'    → 평균 active 페어 {avg_active:.1f}개  '
          f'블락 총 {n_blocked_total}회 ({n_blocked_unique}개 페어)')


# ============================================================
# 4. 종합 테이블 + 단조성 검증
# ============================================================
print()
print('=' * 72)
print('4. 종합 비교')
print('=' * 72)
print()
print(f'  {"Cooldown":>8s} {"최종자본":>11s} {"수익률":>9s} {"Sharpe":>7s} '
      f'{"MDD":>8s} {"평균페어":>8s} {"블락회수":>8s}')
print(f'  {"-"*8} {"-"*11} {"-"*9} {"-"*7} {"-"*8} {"-"*8} {"-"*8}')
for v in variants:
    print(f'  {v["cd"]:>6d}일 '
          f'${v["final"]:>9,.0f}  '
          f'{v["return"]:>+8.2%} '
          f'{v["sharpe"]:>+7.2f} '
          f'{v["mdd"]:>+7.2%} '
          f'{v["avg_active_pairs"]:>8.1f} '
          f'{v["n_blocked_total"]:>8d}')

# 참조 기준
print(f'  ─────── 참조 ───────')
ret_b_pct = ret_b * 100
sh_b = equity_b.pct_change().dropna()
sharpe_b = sh_b.mean() / sh_b.std() * np.sqrt(252) if sh_b.std() > 0 else 0
mdd_b = (equity_b / equity_b.cummax() - 1).min()
print(f'  {"(b) 고정":>8s} '
      f'${equity_b.iloc[-1]:>9,.0f}  '
      f'{ret_b:>+8.2%} '
      f'{sharpe_b:>+7.2f} '
      f'{mdd_b:>+7.2%} '
      f'{len(fixed_pool):>8d} '
      f'{"-":>8s}')

# 단조성 점검
returns_seq = [v['return'] for v in variants]
sharpes_seq = [v['sharpe'] for v in variants]

ret_diffs = np.diff(returns_seq)
sh_diffs = np.diff(sharpes_seq)
ret_monotone = all(d >= 0 for d in ret_diffs) or all(d <= 0 for d in ret_diffs)
sh_monotone = all(d >= 0 for d in sh_diffs) or all(d <= 0 for d in sh_diffs)

print()
print(f'  단조성 점검:')
print(f'    수익률 (cd 0→90): {returns_seq}')
print(f'    Sharpe (cd 0→90): {sharpes_seq}')
print(f'    수익률 단조? {ret_monotone}')
print(f'    Sharpe 단조?  {sh_monotone}')

# 최적값
best_ret_idx = int(np.argmax(returns_seq))
best_sh_idx = int(np.argmax(sharpes_seq))
print()
print(f'  최고 수익률: cooldown={cooldowns[best_ret_idx]}일 '
      f'({returns_seq[best_ret_idx]:+.2%})')
print(f'  최고 Sharpe : cooldown={cooldowns[best_sh_idx]}일 '
      f'({sharpes_seq[best_sh_idx]:+.2f})')

# 양수 회복 여부
recovered = [v for v in variants if v['return'] > 0]
if recovered:
    rec_str = ", ".join(f"cd={v['cd']}" for v in recovered)
    print(f'  ★ 양수 회복 변형: {rec_str}')
else:
    print(f'  ✗ 모든 cooldown 변형이 여전히 음수 수익')

# (b) 도달 여부
beat_b = [v for v in variants if v['return'] >= ret_b]
if beat_b:
    beat_str = ", ".join(f"cd={v['cd']}" for v in beat_b)
    print(f'  ★ (b) 기준선 도달: {beat_str}')
else:
    print(f'  ✗ 어떤 cooldown도 (b) +{ret_b*100:.2f}%에 도달 못 함')


# ============================================================
# 5. 시각화
# ============================================================
print()
print('=' * 72)
print('5. 시각화')
print('=' * 72)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.28)

# (1, full row) 자본 곡선
ax = fig.add_subplot(gs[0, :])
cmap = plt.cm.viridis
for i, v in enumerate(variants):
    color = cmap(i / (len(variants) - 1))
    ax.plot(v['equity'].index, v['equity'],
            label=f'cooldown={v["cd"]}d  → ${v["final"]:,.0f} ({v["return"]:+.2%})',
            color=color, lw=1.8 if v['cd'] == 30 else 1.3,
            alpha=1.0 if v['cd'] == 30 else 0.85)
ax.plot(equity_b.index, equity_b,
        label=f'(b) Fixed pool reference  → ${equity_b.iloc[-1]:,.0f} ({ret_b:+.2%})',
        color='red', lw=2.0, ls='--')
ax.axhline(INITIAL_CAPITAL, color='black', ls=':', alpha=0.5)
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title('MVP 9-B: Equity curves — cooldown sweep')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) Return + Sharpe vs cooldown
ax = fig.add_subplot(gs[1, 0])
ax2 = ax.twinx()
cds = [v['cd'] for v in variants]
rets = [v['return'] * 100 for v in variants]
shs = [v['sharpe'] for v in variants]

ax.plot(cds, rets, 'o-', color='steelblue', lw=2, markersize=10,
        label='Return %')
ax2.plot(cds, shs, 's--', color='darkorange', lw=2, markersize=10,
         label='Sharpe')
ax.axhline(0, color='black', lw=0.5)
ax.axhline(ret_b * 100, color='red', ls=':', alpha=0.7,
           label=f'(b) ref {ret_b*100:+.2f}%')
ax.set_xlabel('Cooldown days')
ax.set_ylabel('Total Return (%)', color='steelblue')
ax2.set_ylabel('Sharpe', color='darkorange')
ax.set_title('Cooldown vs performance — sweep')
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(cds)

# (2, 2) MDD vs cooldown
ax = fig.add_subplot(gs[1, 1])
mdds = [v['mdd'] * 100 for v in variants]
colors_seq = [cmap(i / (len(variants) - 1)) for i in range(len(variants))]
bars = ax.bar([str(c) for c in cds], mdds, color=colors_seq, alpha=0.85)
ax.axhline(mdd_b * 100, color='red', ls=':', alpha=0.7,
           label=f'(b) ref {mdd_b*100:.2f}%')
ax.set_xlabel('Cooldown days')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown vs cooldown')
for bar, val in zip(bars, mdds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val - 0.5, f'{val:.1f}%',
            ha='center', va='top', fontsize=9)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 1) 평균 active 페어 vs cooldown
ax = fig.add_subplot(gs[2, 0])
avgs = [v['avg_active_pairs'] for v in variants]
blockeds = [v['n_blocked_total'] for v in variants]

ax2 = ax.twinx()
ax.bar([c - 1.5 for c in cds], avgs, width=3, color='steelblue', alpha=0.8,
       label='Avg active pairs')
ax2.bar([c + 1.5 for c in cds], blockeds, width=3, color='red', alpha=0.6,
        label='Blocked total')
ax.set_xlabel('Cooldown days')
ax.set_ylabel('Avg active pairs per month', color='steelblue')
ax2.set_ylabel('Total blocked-by-cooldown events', color='red')
ax.set_title('Cost of cooldown: fewer active pairs')
ax.set_xticks(cds)
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 2) Drawdown curves
ax = fig.add_subplot(gs[2, 1])
for i, v in enumerate(variants):
    color = cmap(i / (len(variants) - 1))
    dd = (v['equity'] / v['equity'].cummax() - 1) * 100
    ax.plot(dd.index, dd, label=f'cd={v["cd"]}d', color=color,
            lw=1.8 if v['cd'] == 30 else 1.0,
            alpha=1.0 if v['cd'] == 30 else 0.7)
dd_b = (equity_b / equity_b.cummax() - 1) * 100
ax.plot(dd_b.index, dd_b, color='red', ls='--', lw=1.8, label='(b) ref')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.set_title('Drawdown comparison')
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

plt.suptitle('MVP 9-B: Cooldown sweep for new pairs', fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9b_cooldown_sweep.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'   차트 저장: {out_path}')


# ============================================================
# 6. 결론
# ============================================================
print()
print('=' * 72)
print('6. 결론')
print('=' * 72)
print()
print(f'  9차-B 가설: "cooldown=30일이면 양수 회복"')
v30 = next(v for v in variants if v['cd'] == 30)
print(f'    결과: cd=30 수익률 {v30["return"]:+.2%}, Sharpe {v30["sharpe"]:+.2f}')
if v30['return'] > 0:
    print(f'    ✓ 양수 회복 입증')
else:
    print(f'    ✗ 양수 회복 실패 — but baseline 대비 변화 분석 필요')

v0 = next(v for v in variants if v['cd'] == 0)
delta_ret = v30['return'] - v0['return']
delta_sh = v30['sharpe'] - v0['sharpe']
print()
print(f'  cd=30 vs cd=0:')
print(f'    수익률: {v0["return"]:+.2%} → {v30["return"]:+.2%} ({delta_ret:+.2%}p)')
print(f'    Sharpe: {v0["sharpe"]:+.2f} → {v30["sharpe"]:+.2f} ({delta_sh:+.2f})')

# 결과 dump
import json
report_data = {
    'cooldown_sweep': [
        {'cd': v['cd'], 'final': v['final'], 'return': v['return'],
         'sharpe': v['sharpe'], 'mdd': v['mdd'],
         'avg_pairs': v['avg_active_pairs'],
         'n_blocked': v['n_blocked_total']}
        for v in variants
    ],
    'reference_b': {
        'return': ret_b,
        'sharpe': float(sharpe_b),
        'mdd': float(mdd_b),
    },
    'monotonicity': {
        'return_monotone': bool(ret_monotone),
        'sharpe_monotone': bool(sh_monotone),
    },
    'best_cooldown': {
        'by_return': cooldowns[best_ret_idx],
        'by_sharpe': cooldowns[best_sh_idx],
    },
    'recovered_positive': [v['cd'] for v in variants if v['return'] > 0],
    'beat_b': [v['cd'] for v in variants if v['return'] >= ret_b],
}
data_dump_path = '/tmp/mvp9b_report_data.json'
with open(data_dump_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print()
print(f'  데이터 저장: {data_dump_path}')
