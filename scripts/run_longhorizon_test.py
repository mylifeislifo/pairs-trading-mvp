"""
9차 MVP-E: 장기 데이터로 9-D 결과 재검증
============================================

9-D 결과의 통계적 견고성 점검:
  - 3년 데이터에서 refresh=180은 snapshot 3개뿐
  - (refresh=180, cd=30) Sharpe +0.73이 진짜 alpha인지 / 표본 부족 인공물인지

검증 방법:
  1. 10년 데이터로 동일 grid 재실행 (snapshot 5-6배 증가)
  2. 시간 분할 (전반 5년 / 후반 5년)로 안정성 확인
  3. (b) 참조도 같은 기간으로 재산출

검증 변형 (6개):
  (b) 고정 풀
  (r30, cd=0)  ← 7차 baseline, 장기에서 더 망하는지
  (r30, cd=30) ← 9-B 보수적 회복안
  (r90, cd=30) ← 9-D "통계적으로 안전한 추천"
  (r180, cd=0) ← 9-D Return 1위
  (r180, cd=30)← 9-D Sharpe 1위

3가지 시나리오 판정:
  ① 9-D 결과 견고: 장기에서도 같은 변형이 (b)에 근접/능가
  ② 9-D 결과 인공물: 장기에서 결과 무너짐 → 9-D는 운빨
  ③ 부분 견고: 일부만 살아남음
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
# 1. 데이터 — 10년
# ============================================================
TICKERS = [
    'KO', 'PEP', 'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX', 'V', 'MA', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T', 'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW', 'WMT', 'TGT',
]

print('=' * 72)
print('9차 MVP-E: 10년 장기 데이터로 9-D 결과 재검증')
print('=' * 72)
print()
print('1. 데이터 다운로드 (10년)')
data = yf.download(TICKERS, period='10y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'   {data.shape[0]}일 × {data.shape[1]}종목')
print(f'   기간: {data.index[0].date()} ~ {data.index[-1].date()}')

INITIAL_CAPITAL = 100_000
finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)


# ============================================================
# 헬퍼: equity curve로부터 metrics 계산
# ============================================================
def calc_metrics(eq):
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return ret, sharpe, mdd


def calc_metrics_slice(eq, start_date, end_date):
    """시간 구간별 metrics"""
    slc = eq.loc[start_date:end_date]
    if len(slc) < 10:
        return None, None, None
    ret = slc.iloc[-1] / slc.iloc[0] - 1
    daily = slc.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((slc / slc.cummax() - 1).min())
    return ret, sharpe, mdd


# ============================================================
# 2. (b) 재현 — 10년 데이터로
# ============================================================
print()
print('2. (b) 고정 풀 재현 — 10년 train_split 70%')

train_split = int(len(data) * 0.7)
print(f'   Train 기간: {data.index[0].date()} ~ {data.index[train_split-1].date()}')

fixed_pool = finder.screen_pairs(data.iloc[:train_split])
print(f'   발굴 페어 수: {len(fixed_pool)}개')
for p in fixed_pool[:10]:
    print(f'     {p}')
if len(fixed_pool) > 10:
    print(f'     ... +{len(fixed_pool)-10}개 더')

daily_returns_b = pd.Series(0.0, index=data.index)
equal_w = 0.10 / max(len(fixed_pool), 1)
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
ret_b, sharpe_b, mdd_b = calc_metrics(equity_b)
print(f'   (b) 최종 ${equity_b.iloc[-1]:,.0f} ({ret_b:+.2%}), '
      f'Sharpe {sharpe_b:+.2f}, MDD {mdd_b:+.2%}')


# ============================================================
# 3. 6개 변형 백테스트
# ============================================================
print()
print('=' * 72)
print('3. 6개 핵심 변형 — 10년 백테스트')
print('=' * 72)

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

variants = [
    ('r30, cd=0',  {'refresh_every_days': 30,  'pair_cooldown_days': 0}),
    ('r30, cd=30', {'refresh_every_days': 30,  'pair_cooldown_days': 30}),
    ('r90, cd=30', {'refresh_every_days': 90,  'pair_cooldown_days': 30}),
    ('r180, cd=0', {'refresh_every_days': 180, 'pair_cooldown_days': 0}),
    ('r180, cd=30',{'refresh_every_days': 180, 'pair_cooldown_days': 30}),
]

results = {}
for label, params in variants:
    print()
    print(f'  {label} ...', end=' ', flush=True)
    sys_v = ProductionSystem(**common_kwargs, **params)
    res = sys_v.run(data, verbose=False)
    eq = res.equity_curve
    ret, sh, mdd = calc_metrics(eq)
    n_snapshots = len(res.monthly_states)
    avg_pairs = np.mean([len(s.active_pairs) for s in res.monthly_states
                         if s.active_pairs]) if any(s.active_pairs for s in res.monthly_states) else 0
    # 회전율
    prev_pids = set()
    n_new = 0
    for st in res.monthly_states:
        cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
        n_new += len(cur_pids - prev_pids)
        prev_pids = cur_pids

    results[label] = {
        'equity': eq, 'return': ret, 'sharpe': sh, 'mdd': mdd,
        'n_snapshots': n_snapshots, 'avg_pairs': avg_pairs, 'n_new': n_new,
        'params': params,
    }
    print(f'final ${eq.iloc[-1]:>9,.0f}  Ret {ret:+7.2%}  Sh {sh:+5.2f}  '
          f'MDD {mdd:+6.2%}  snap {n_snapshots} pairs {avg_pairs:.1f}')


# ============================================================
# 4. 종합 테이블
# ============================================================
print()
print('=' * 72)
print('4. 10년 종합 결과')
print('=' * 72)
print()
print(f'  {"Variant":<15s} {"최종":>11s} {"Return":>9s} {"Sharpe":>7s} '
      f'{"MDD":>8s} {"snap":>5s} {"pairs":>6s} {"new":>5s}')
print(f'  {"-"*15} {"-"*11} {"-"*9} {"-"*7} {"-"*8} {"-"*5} {"-"*6} {"-"*5}')

for label in [l for l, _ in variants]:
    r = results[label]
    print(f'  {label:<15s} '
          f'${r["equity"].iloc[-1]:>9,.0f}  '
          f'{r["return"]:>+8.2%} '
          f'{r["sharpe"]:>+7.2f} '
          f'{r["mdd"]:>+7.2%} '
          f'{r["n_snapshots"]:>5d} '
          f'{r["avg_pairs"]:>6.1f} '
          f'{r["n_new"]:>5d}')

print(f'  {"-"*15}')
print(f'  {"(b) ref":<15s} '
      f'${equity_b.iloc[-1]:>9,.0f}  '
      f'{ret_b:>+8.2%} '
      f'{sharpe_b:>+7.2f} '
      f'{mdd_b:>+7.2%} '
      f'{"-":>5s} '
      f'{len(fixed_pool):>6d} '
      f'{"-":>5s}')


# ============================================================
# 5. 시간 분할 안정성 — 전반/후반 5년
# ============================================================
print()
print('=' * 72)
print('5. 시간 분할 안정성 — 전반 5년 / 후반 5년')
print('=' * 72)

mid_date = data.index[len(data) // 2]
end_date = data.index[-1]
start_date = data.index[0]
print(f'   전반: {start_date.date()} ~ {mid_date.date()}')
print(f'   후반: {mid_date.date()} ~ {end_date.date()}')
print()

# 각 변형별 전반/후반 metrics
print(f'  {"Variant":<15s} {"전반 Ret":>10s} {"후반 Ret":>10s} {"전반 Sh":>9s} '
      f'{"후반 Sh":>9s} {"안정성":>9s}')
print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*9} {"-"*9} {"-"*9}')

stability_data = {}
for label in [l for l, _ in variants]:
    eq = results[label]['equity']
    ret_1, sh_1, mdd_1 = calc_metrics_slice(eq, start_date, mid_date)
    ret_2, sh_2, mdd_2 = calc_metrics_slice(eq, mid_date, end_date)
    # 안정성: 두 구간 Sharpe 부호 일치 + 크기 비슷
    sign_match = (sh_1 > 0) == (sh_2 > 0) if sh_1 is not None and sh_2 is not None else False
    stability = '✓ 안정' if sign_match else '✗ 불안정'
    stability_data[label] = {
        'first_half': {'ret': ret_1, 'sharpe': sh_1, 'mdd': mdd_1},
        'second_half': {'ret': ret_2, 'sharpe': sh_2, 'mdd': mdd_2},
        'stable': sign_match,
    }
    print(f'  {label:<15s} '
          f'{ret_1*100 if ret_1 is not None else 0:>+9.2f}% '
          f'{ret_2*100 if ret_2 is not None else 0:>+9.2f}% '
          f'{sh_1 if sh_1 is not None else 0:>+8.2f} '
          f'{sh_2 if sh_2 is not None else 0:>+8.2f} '
          f'{stability:>9s}')

# (b)도 추가
ret_1b, sh_1b, _ = calc_metrics_slice(equity_b, start_date, mid_date)
ret_2b, sh_2b, _ = calc_metrics_slice(equity_b, mid_date, end_date)
sign_match_b = (sh_1b > 0) == (sh_2b > 0)
print(f'  {"(b) ref":<15s} '
      f'{ret_1b*100:>+9.2f}% '
      f'{ret_2b*100:>+9.2f}% '
      f'{sh_1b:>+8.2f} '
      f'{sh_2b:>+8.2f} '
      f'{"✓ 안정" if sign_match_b else "✗ 불안정":>9s}')
stability_data['(b)'] = {
    'first_half': {'ret': ret_1b, 'sharpe': sh_1b},
    'second_half': {'ret': ret_2b, 'sharpe': sh_2b},
    'stable': sign_match_b,
}


# ============================================================
# 6. 3년 vs 10년 비교 — 9-D 견고성 판정
# ============================================================
print()
print('=' * 72)
print('6. 3년(9-D) vs 10년(9-E) 결과 비교 — 견고성 판정')
print('=' * 72)

# 9-D 결과 (3년 백테스트)
results_3yr = {
    'r30, cd=0':   {'ret': -0.0653, 'sharpe': -0.45},
    'r30, cd=30':  {'ret': +0.0023, 'sharpe': +0.04},
    'r90, cd=30':  {'ret': +0.0267, 'sharpe': +0.48},
    'r180, cd=0':  {'ret': +0.0654, 'sharpe': +0.32},
    'r180, cd=30': {'ret': +0.0293, 'sharpe': +0.73},
}
ret_b_3yr = 0.0294
sharpe_b_3yr = +0.83

print()
print(f'  {"Variant":<15s} {"3년 Sh":>8s} {"10년 Sh":>9s} {"Δ Sh":>8s} {"3년 Ret":>9s} {"10년 Ret":>10s} {"판정":>10s}')
print(f'  {"-"*15} {"-"*8} {"-"*9} {"-"*8} {"-"*9} {"-"*10} {"-"*10}')

for label, _ in variants:
    r_3 = results_3yr[label]
    r_10 = results[label]
    # 연환산으로 비교 (단순 Sharpe는 이미 연환산)
    delta_sh = r_10['sharpe'] - r_3['sharpe']
    sign_match = (r_3['sharpe'] > 0) == (r_10['sharpe'] > 0)
    verdict = '✓ 견고' if sign_match else '✗ 무너짐'
    print(f'  {label:<15s} '
          f'{r_3["sharpe"]:>+7.2f} '
          f'{r_10["sharpe"]:>+8.2f} '
          f'{delta_sh:>+7.2f} '
          f'{r_3["ret"]*100:>+8.2f}% '
          f'{r_10["return"]*100:>+9.2f}% '
          f'{verdict:>10s}')

print(f'  {"(b) ref":<15s} '
      f'{sharpe_b_3yr:>+7.2f} '
      f'{sharpe_b:>+8.2f} '
      f'{sharpe_b-sharpe_b_3yr:>+7.2f} '
      f'{ret_b_3yr*100:>+8.2f}% '
      f'{ret_b*100:>+9.2f}% '
      f'{"-":>10s}')


# ============================================================
# 7. 시각화
# ============================================================
print()
print('=' * 72)
print('7. 시각화')
print('=' * 72)

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full row) Equity curves (long-term)
ax = fig.add_subplot(gs[0, :])
color_map = {
    'r30, cd=0':   'red',
    'r30, cd=30':  'orange',
    'r90, cd=30':  'darkgreen',
    'r180, cd=0':  'darkorange',
    'r180, cd=30': 'purple',
}
for label in [l for l, _ in variants]:
    r = results[label]
    ax.plot(r['equity'].index, r['equity'],
            label=f'{label} → {r["return"]:+.2%} (Sh {r["sharpe"]:+.2f})',
            color=color_map[label], lw=1.5)
ax.plot(equity_b.index, equity_b,
        label=f'(b) Fixed ref → {ret_b:+.2%} (Sh {sharpe_b:+.2f})',
        color='black', lw=2.2, ls='--')
ax.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.5)
ax.axvline(mid_date, color='red', ls=':', alpha=0.4, label='Split midpoint')
ax.set_ylabel('Equity (USD, log scale)')
ax.set_yscale('log')
ax.set_xlabel('Date')
ax.set_title(f'MVP 9-E: 10-year equity curves '
             f'({data.index[0].date()} ~ {data.index[-1].date()})')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

# (2, 1) 3년 vs 10년 Sharpe 비교
ax = fig.add_subplot(gs[1, 0])
labels_short = [l for l, _ in variants] + ['(b)']
sharpes_3yr = [results_3yr[l]['sharpe'] for l, _ in variants] + [sharpe_b_3yr]
sharpes_10yr = [results[l]['sharpe'] for l, _ in variants] + [sharpe_b]

x = np.arange(len(labels_short))
w = 0.35
ax.bar(x - w/2, sharpes_3yr, w, color='steelblue', alpha=0.85, label='3-year (9-D)')
ax.bar(x + w/2, sharpes_10yr, w, color='darkorange', alpha=0.85, label='10-year (9-E)')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Sharpe: 3-year vs 10-year — robustness check')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (2, 2) 전반 / 후반 5년 Sharpe (시간 분할 안정성)
ax = fig.add_subplot(gs[1, 1])
sharpes_first = [stability_data[l]['first_half']['sharpe'] for l, _ in variants] + [stability_data['(b)']['first_half']['sharpe']]
sharpes_second = [stability_data[l]['second_half']['sharpe'] for l, _ in variants] + [stability_data['(b)']['second_half']['sharpe']]

ax.bar(x - w/2, sharpes_first, w, color='lightblue', alpha=0.85, label='First half')
ax.bar(x + w/2, sharpes_second, w, color='lightcoral', alpha=0.85, label='Second half')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Time-split stability — first half vs second half (5 years each)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 1) Return % + MDD heatmap (10-yr)
ax = fig.add_subplot(gs[2, 0])
returns_10 = [results[l]['return']*100 for l, _ in variants] + [ret_b*100]
mdds_10 = [results[l]['mdd']*100 for l, _ in variants] + [mdd_b*100]

ax2 = ax.twinx()
ax.bar(x - w/2, returns_10, w,
       color=[color_map[l] for l, _ in variants] + ['black'],
       alpha=0.85, label='Total Return %')
ax2.bar(x + w/2, mdds_10, w,
        color=[color_map[l] for l, _ in variants] + ['black'],
        alpha=0.45, edgecolor='black', label='MDD %')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Total Return % (solid)')
ax2.set_ylabel('Max Drawdown % (faded)')
ax.set_title('10-year Return vs MDD')
ax.grid(alpha=0.3, axis='y')

# (3, 2) Drawdown curves
ax = fig.add_subplot(gs[2, 1])
for label in [l for l, _ in variants]:
    r = results[label]
    dd = (r['equity'] / r['equity'].cummax() - 1) * 100
    ax.plot(dd.index, dd, label=label, color=color_map[label], lw=1.2, alpha=0.85)
dd_b = (equity_b / equity_b.cummax() - 1) * 100
ax.plot(dd_b.index, dd_b, color='black', lw=2, ls='--', label='(b) ref')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.set_title('Long-term drawdown — depth & duration')
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

plt.suptitle('MVP 9-E: 10-year validation of 9-D findings', fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9e_long_horizon.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'   차트 저장: {out_path}')


# ============================================================
# 8. 결론
# ============================================================
print()
print('=' * 72)
print('8. 핵심 결론')
print('=' * 72)
print()

# 견고성 판정
print('  견고성 판정 (3년 vs 10년):')
all_robust = True
for label, _ in variants:
    r_3 = results_3yr[label]['sharpe']
    r_10 = results[label]['sharpe']
    sign_match = (r_3 > 0) == (r_10 > 0)
    if not sign_match:
        all_robust = False
        print(f'    {label}: ✗ 부호 반전 ({r_3:+.2f} → {r_10:+.2f})')

# 최고 변형 (10년)
best_label = max(results, key=lambda k: results[k]['sharpe'])
best_r = results[best_label]
print()
print(f'  10년 최고 Sharpe: {best_label}')
print(f'    Return {best_r["return"]:+.2%}, Sharpe {best_r["sharpe"]:+.2f}, MDD {best_r["mdd"]:+.2%}')

# (b)와 비교
beat_b_count = sum(1 for l, _ in variants if results[l]['sharpe'] >= sharpe_b)
print()
print(f'  (b) Sharpe {sharpe_b:+.2f} 능가/동등: {beat_b_count}개 변형')

# 데이터 dump
import json
report_data = {
    'data_period': {
        'start': str(data.index[0].date()),
        'end': str(data.index[-1].date()),
        'days': int(len(data)),
        'tickers': int(data.shape[1]),
    },
    'reference_b_10yr': {
        'return': ret_b, 'sharpe': float(sharpe_b), 'mdd': float(mdd_b),
        'fixed_pool_size': len(fixed_pool),
    },
    'variants_10yr': {
        label: {
            'return': r['return'], 'sharpe': r['sharpe'], 'mdd': r['mdd'],
            'n_snapshots': r['n_snapshots'], 'avg_pairs': r['avg_pairs'],
            'n_new': r['n_new'],
        }
        for label, r in results.items()
    },
    'stability_check': {
        label: {
            'first_half_sharpe': s['first_half']['sharpe'],
            'second_half_sharpe': s['second_half']['sharpe'],
            'stable': s['stable'],
        }
        for label, s in stability_data.items()
    },
    'comparison_3yr_vs_10yr': {
        label: {
            '3yr_sharpe': results_3yr[label]['sharpe'],
            '10yr_sharpe': results[label]['sharpe'],
            'sign_preserved': (results_3yr[label]['sharpe'] > 0) == (results[label]['sharpe'] > 0),
        }
        for label, _ in variants
    },
    'all_robust': bool(all_robust),
    'best_variant_10yr': best_label,
}
data_dump_path = '/tmp/mvp9e_report_data.json'
with open(data_dump_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print()
print(f'  데이터 저장: {data_dump_path}')
