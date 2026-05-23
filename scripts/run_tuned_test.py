"""
8차 MVP: 페어 품질 필터링 튜닝 비교
========================================

7차 발견 — 완전 통합 시스템 (d)이 -20.86%로 망한 이유:
  소수의 악성 페어(AMD~C: -$4,098, AMD~T: -$3,767, AMD~NVDA: -$2,507)가
  좋은 페어 10개의 수익을 다 까먹음.

8차 가설:
  "max_active_pairs=5 + min_historical_sharpe=0 적용하면 회복된다"

5-way 비교:
  (d)  baseline       : 7차 그대로 (필터 없음)
  (d1) cap=5 only     : 페어 수만 제한 (Sharpe 무관)
  (d2) min_sharpe=0   : Sharpe 음수 페어만 차단
  (d3) cap=5 + min=0  : 둘 다 적용 — 8차 최종 후보
  (d4) cap=5 + min=0.5: 공격적 필터 (Sharpe 0.5 이상만)

평가:
  - 최종 자본, Sharpe, MDD
  - 매월 운용 페어 수 분포
  - 필터링이 어떤 페어를 잘라냈는가
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

# src/ 경로 등록
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pairs_trading_mvp import PairsFinder
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
print('8차 MVP: 페어 품질 필터링 튜닝 — 5-way 비교')
print('=' * 72)
print()
print('1. 데이터 다운로드')
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'   {data.shape[0]}일 × {data.shape[1]}종목')

INITIAL_CAPITAL = 100_000


# ============================================================
# 2. 5-way 비교 백테스트
# ============================================================
print()
print('=' * 72)
print('2. 5가지 튜닝 변형 백테스트')
print('=' * 72)

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)

# 공통 파라미터
common_kwargs = dict(
    finder=finder,
    lookback_days=365,
    refresh_every_days=30,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=INITIAL_CAPITAL,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=True,
    quality_lookback=90,
)

variants = [
    {
        'key': '(d) baseline (7th MVP)',
        'short': '(d)',
        'desc': '필터 없음 — 7차 그대로',
        'params': {**common_kwargs,
                   'max_active_pairs': None,
                   'min_historical_sharpe': None},
        'color': 'red',
    },
    {
        'key': '(d1) cap=5 only',
        'short': '(d1)',
        'desc': '페어 수만 5개로 cap (Sharpe 무관)',
        'params': {**common_kwargs,
                   'max_active_pairs': 5,
                   'min_historical_sharpe': None},
        'color': 'orange',
    },
    {
        'key': '(d2) min_sharpe=0 only',
        'short': '(d2)',
        'desc': '과거 Sharpe 음수 페어만 차단',
        'params': {**common_kwargs,
                   'max_active_pairs': None,
                   'min_historical_sharpe': 0.0},
        'color': 'steelblue',
    },
    {
        'key': '(d3) cap=5 + min_sharpe=0',
        'short': '(d3)',
        'desc': '8차 최종 후보 — 둘 다 적용',
        'params': {**common_kwargs,
                   'max_active_pairs': 5,
                   'min_historical_sharpe': 0.0},
        'color': 'darkgreen',
    },
    {
        'key': '(d4) cap=5 + min_sharpe=0.5',
        'short': '(d4)',
        'desc': '공격적 — Sharpe 0.5 이상만 운용',
        'params': {**common_kwargs,
                   'max_active_pairs': 5,
                   'min_historical_sharpe': 0.5},
        'color': 'purple',
    },
]

results = {}
for v in variants:
    print()
    print(f'  {v["key"]}')
    print(f'    {v["desc"]}')
    sys_v = ProductionSystem(**v['params'])
    res = sys_v.run(data, verbose=False)
    results[v['key']] = {
        'system': sys_v,
        'result': res,
        'meta': v,
    }
    m = res.metrics
    print(f'    → 최종 ${res.equity_curve.iloc[-1]:>10,.0f}  '
          f'Return {m["total_return"]:+7.2%}  '
          f'Sharpe {m["sharpe"]:+5.2f}  '
          f'MDD {m["max_drawdown"]:+6.2%}')


# ============================================================
# 3. 종합 비교 테이블
# ============================================================
print()
print('=' * 72)
print('3. 종합 비교')
print('=' * 72)
print()
print(f'  {"변형":<32s} {"최종자본":>11s} {"수익률":>9s} {"Sharpe":>7s} '
      f'{"MDD":>8s} {"Calmar":>7s} {"평균페어":>8s}')
print(f'  {"-"*32} {"-"*11} {"-"*9} {"-"*7} {"-"*8} {"-"*7} {"-"*8}')

summary_rows = []
for v in variants:
    key = v['key']
    res = results[key]['result']
    m = res.metrics
    # 평균 active pairs per month
    pair_counts = [len(s.active_pairs) for s in res.monthly_states
                   if s.active_pairs]
    avg_pairs = np.mean(pair_counts) if pair_counts else 0
    calmar = m.get('calmar', 0)
    if calmar == np.inf:
        calmar_s = '   inf'
    else:
        calmar_s = f'{calmar:>+6.2f}'
    print(f'  {key:<32s} '
          f'${res.equity_curve.iloc[-1]:>9,.0f}  '
          f'{m["total_return"]:>+8.2%} '
          f'{m["sharpe"]:>+7.2f} '
          f'{m["max_drawdown"]:>+7.2%} '
          f'{calmar_s} '
          f'{avg_pairs:>8.1f}')
    summary_rows.append({
        'variant': v['short'],
        'key': key,
        'final': float(res.equity_curve.iloc[-1]),
        'total_return': m['total_return'],
        'sharpe': m['sharpe'],
        'mdd': m['max_drawdown'],
        'calmar': calmar,
        'avg_pairs': avg_pairs,
    })


# ============================================================
# 4. 필터링 진단 — 어떤 페어가 잘려나갔는가
# ============================================================
print()
print('=' * 72)
print('4. 필터링 진단 (d3 기준)')
print('=' * 72)
print()

d3_system = results['(d3) cap=5 + min_sharpe=0']['system']
filter_log = d3_system.filter_log

if filter_log:
    avg_candidates = np.mean([e['candidates'] for e in filter_log])
    avg_after_sharpe = np.mean([e['after_sharpe_filter'] for e in filter_log])
    avg_final = np.mean([e['final'] for e in filter_log])
    print(f'  매월 평균 후보 페어   : {avg_candidates:.1f}개')
    print(f'  Sharpe 필터 통과     : {avg_after_sharpe:.1f}개')
    print(f'  cap=5 적용 후 최종    : {avg_final:.1f}개')
    print()

    # 가장 자주 컷된 페어들
    from collections import Counter
    cut_counter = Counter()
    select_counter = Counter()
    for e in filter_log:
        for pid in e['pair_sharpes']:
            if pid not in e['selected']:
                cut_counter[pid] += 1
            else:
                select_counter[pid] += 1

    print('  ▷ 가장 자주 선택된 페어 (상위 10개):')
    for pid, cnt in select_counter.most_common(10):
        avg_sh = np.mean([e['pair_sharpes'].get(pid, 0)
                          for e in filter_log
                          if pid in e['pair_sharpes']])
        print(f'    {pid:<18s}  선택 {cnt}회  (평균 Sharpe {avg_sh:+.2f})')

    print()
    print('  ▷ 가장 자주 컷된 페어 (상위 10개):')
    for pid, cnt in cut_counter.most_common(10):
        avg_sh = np.mean([e['pair_sharpes'].get(pid, 0)
                          for e in filter_log
                          if pid in e['pair_sharpes']])
        print(f'    {pid:<18s}  컷 {cnt}회   (평균 Sharpe {avg_sh:+.2f})')


# ============================================================
# 5. 시각화 — 차트 라벨은 영어
# ============================================================
print()
print('=' * 72)
print('5. 시각화')
print('=' * 72)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.28)

# (1, full row) 자본 곡선 5-way 비교
ax = fig.add_subplot(gs[0, :])
for v in variants:
    key = v['key']
    eq = results[key]['result'].equity_curve
    is_d3 = '(d3)' in key
    ax.plot(eq.index, eq, label=key, color=v['color'],
            lw=2.0 if is_d3 else 1.3,
            alpha=1.0 if is_d3 else 0.75)
ax.axhline(INITIAL_CAPITAL, color='black', ls=':', alpha=0.5,
           label=f'Initial capital ${INITIAL_CAPITAL:,}')
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title('MVP 8: Equity curves — 5 tuning variants')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) Sharpe + Return 막대 비교
ax = fig.add_subplot(gs[1, 0])
labels = [v['short'] for v in variants]
sharpes = [r['sharpe'] for r in summary_rows]
returns_pct = [r['total_return'] * 100 for r in summary_rows]
colors_seq = [v['color'] for v in variants]

x = np.arange(len(labels))
ax2 = ax.twinx()
ax.bar(x - 0.2, sharpes, 0.4, color=colors_seq, alpha=0.85, label='Sharpe')
ax2.bar(x + 0.2, returns_pct, 0.4, color=colors_seq,
        alpha=0.45, edgecolor='black', label='Total Return %')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Sharpe ratio (left, solid)')
ax2.set_ylabel('Total Return % (right, faded)')
ax.set_title('Sharpe vs Total Return')
ax.grid(alpha=0.3, axis='y')

# (2, 2) MDD 비교
ax = fig.add_subplot(gs[1, 1])
mdds_pct = [r['mdd'] * 100 for r in summary_rows]
bars = ax.bar(x, mdds_pct, color=colors_seq, alpha=0.85)
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown — lower (more negative) is worse')
for bar, val in zip(bars, mdds_pct):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val - 1, f'{val:.1f}%',
            ha='center', va='top', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 1) Drawdown curve
ax = fig.add_subplot(gs[2, 0])
for v in variants:
    key = v['key']
    eq = results[key]['result'].equity_curve
    dd = (eq / eq.cummax() - 1) * 100
    is_d3 = '(d3)' in key
    ax.plot(dd.index, dd, label=v['short'], color=v['color'],
            lw=1.8 if is_d3 else 1.0,
            alpha=1.0 if is_d3 else 0.7)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.set_title('Drawdown over time')
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (3, 2) Active pairs per month — (d) vs (d3) 비교
ax = fig.add_subplot(gs[2, 1])
res_d = results['(d) baseline (7th MVP)']['result']
res_d3 = results['(d3) cap=5 + min_sharpe=0']['result']
dates_d = [s.date for s in res_d.monthly_states]
n_d = [len(s.active_pairs) for s in res_d.monthly_states]
dates_d3 = [s.date for s in res_d3.monthly_states]
n_d3 = [len(s.active_pairs) for s in res_d3.monthly_states]

w = pd.Timedelta(days=10)
ax.bar([d - w / 2 for d in dates_d], n_d, width=w,
       color='red', alpha=0.5, label='(d) baseline')
ax.bar([d + w / 2 for d in dates_d3], n_d3, width=w,
       color='darkgreen', alpha=0.7, label='(d3) cap=5+min=0')
ax.set_ylabel('Active pairs per month')
ax.set_xlabel('Date')
ax.set_title('Active pairs: baseline vs tuned (d3)')
ax.legend(loc='upper right', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3, axis='y')

plt.suptitle('MVP 8: Quality filter tuning — 5 variants compared',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp8_tuned_comparison.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'   차트 저장: {out_path}')


# ============================================================
# 6. 결론 + 보고서용 데이터 dump
# ============================================================
print()
print('=' * 72)
print('6. 핵심 결론')
print('=' * 72)
print()

# baseline vs 최고
baseline_ret = summary_rows[0]['total_return']
best_idx = int(np.argmax([r['sharpe'] for r in summary_rows]))
best_row = summary_rows[best_idx]
print(f'  (d) baseline 수익률      : {baseline_ret:+.2%}, Sharpe {summary_rows[0]["sharpe"]:+.2f}')
print(f'  최고 Sharpe 변형         : {best_row["key"]} (Sharpe {best_row["sharpe"]:+.2f})')
print()

# 8차 가설 검증
d3_row = next(r for r in summary_rows if '(d3)' in r['key'])
recovered = d3_row['total_return'] > 0
delta = d3_row['total_return'] - baseline_ret
print('  ▷ 8차 가설 검증:')
print(f'    가설: "max=5 + min_sharpe=0이면 양수 회복"')
print(f'    결과: (d3) 수익률 {d3_row["total_return"]:+.2%}, '
      f'baseline 대비 {delta:+.2%}p')
print(f'    가설 {"✓ 입증" if recovered else "✗ 기각"}')

# 7차→8차 교훈
print()
print('  ▷ 핵심 관찰:')
sharpe_improvements = [(r['variant'], r['sharpe'] - summary_rows[0]['sharpe'])
                       for r in summary_rows[1:]]
sharpe_improvements.sort(key=lambda x: x[1], reverse=True)
for vname, dsh in sharpe_improvements:
    direction = '개선' if dsh > 0 else '악화'
    print(f'    {vname}: Sharpe {direction} {dsh:+.2f}')

# 보고서용 데이터 저장
import json
report_data = {
    'summary_rows': summary_rows,
    'd3_filter_stats': {
        'avg_candidates': float(np.mean([e['candidates'] for e in filter_log])) if filter_log else 0,
        'avg_after_sharpe': float(np.mean([e['after_sharpe_filter'] for e in filter_log])) if filter_log else 0,
        'avg_final': float(np.mean([e['final'] for e in filter_log])) if filter_log else 0,
        'top_selected': [(pid, cnt) for pid, cnt in select_counter.most_common(10)] if filter_log else [],
        'top_cut': [(pid, cnt) for pid, cnt in cut_counter.most_common(10)] if filter_log else [],
    },
    'hypothesis_verified': bool(recovered),
}
data_dump_path = '/tmp/mvp8_report_data.json'
with open(data_dump_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print()
print(f'  보고서용 데이터 저장: {data_dump_path}')
