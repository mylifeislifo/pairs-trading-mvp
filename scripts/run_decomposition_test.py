"""
9차 MVP-A: (b) vs (d3) 격차 정밀 분해
=========================================

8차 결과 — 손실 폭은 줄였지만 양수 회복은 못 함:
  (b) 고정 풀     :  +2.94%, Sharpe +0.83  ★
  (d3) 통합+튜닝  :  -6.53%, Sharpe -0.45
  ───────────────────────────────
  격차            :  -9.47%p

가설: "(d3)의 손실 대부분은 페어 갈아타기 비용 + 신규 페어 적응 손실"

4가지 분해:
  A. 거래비용 누적     — (b) vs (d3) 누적 fee+slippage 비교
  B. 페어 회전 횟수    — 풀 진입/퇴장, 평균 페어 수명
  C. 신규 진입 30일 PnL — 페어가 풀에 막 들어왔을 때의 평균 성과
  D. 같은 페어 강제 시  — (d3)에 (b)의 7개 풀을 강제 → 잔여 격차 = "켈리+필터 자체 손실"

(D) 결과 해석:
  - 격차가 사라지면 → "롤링 자체"가 손실의 원인
  - 격차가 남으면   → "켈리 배분/필터" 자체가 균등배분에 비해 손해
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
from collections import defaultdict
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
print('9차 MVP-A: (b) 고정풀 vs (d3) 통합+튜닝 격차 분해')
print('=' * 72)
print()
print('1. 데이터 다운로드')
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'   {data.shape[0]}일 × {data.shape[1]}종목')

INITIAL_CAPITAL = 100_000
FEE = 0.0004
SLIP = 0.0005
COST_PER_ROUND_TRIP = (FEE + SLIP) * 2  # 양쪽 다리 + 진입/청산 = 한 거래당


# ============================================================
# 2. (b) 고정 풀 — 1차 MVP 7개 페어 재현 + trades 수집
# ============================================================
print()
print('=' * 72)
print('2. (b) 고정 풀 백테스트 — 거래 통계 수집')
print('=' * 72)

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)
train_split = int(len(data) * 0.7)
fixed_pool = finder.screen_pairs(data.iloc[:train_split])
print(f'   Train에서 발굴: {len(fixed_pool)}개 페어')
for p in fixed_pool:
    print(f'     {p}')

# 각 페어별 백테스트 + trades 수집
b_pair_results = {}
b_total_cost = 0.0
b_total_trades = 0
daily_returns_b = pd.Series(0.0, index=data.index)
equal_w = 0.10 / len(fixed_pool)

for p in fixed_pool:
    spread = compute_spread(data[p.y], data[p.x], p.beta)
    sw = max(20, min(60, int(p.half_life * 2)))
    sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
    z, pos, fc = sg.generate(spread)
    bt = Backtester(initial_capital=INITIAL_CAPITAL,
                    capital_fraction=1.0,
                    fee_rate=FEE, slippage=SLIP)
    res = bt.run(data[p.y], data[p.x], p.beta, pos, fc)

    pid = f'{p.y}~{p.x}'
    n_trades = len(res.trades)
    # 거래비용 (round-trip 단위): 한 번의 진입→청산 = 2회 포지션 변경
    # 단순화: trades 수 × cost_per_round_trip × pair_weight × initial_capital
    cost = n_trades * COST_PER_ROUND_TRIP * equal_w * INITIAL_CAPITAL
    b_total_cost += cost
    b_total_trades += n_trades

    daily_ret = res.equity.pct_change().fillna(0)
    daily_returns_b += equal_w * daily_ret
    b_pair_results[pid] = {
        'pair': p,
        'trades': res.trades,
        'n_trades': n_trades,
        'cost_estimate': cost,
        'daily_return': daily_ret,
    }
    print(f'     {pid:<18s} trades={n_trades:>3d}  cost≈${cost:>7,.0f}')

equity_b = INITIAL_CAPITAL * (1 + daily_returns_b).cumprod()
print()
print(f'   (b) 총 거래 횟수: {b_total_trades}회')
print(f'   (b) 거래비용 추정: ${b_total_cost:,.0f}')
print(f'   (b) 최종 자본    : ${equity_b.iloc[-1]:,.0f}')


# ============================================================
# 3. (d3) 통합+튜닝 백테스트 — 추가 통계 수집
# ============================================================
print()
print('=' * 72)
print('3. (d3) 통합+튜닝 백테스트 (cap=5 + min_sharpe=0)')
print('=' * 72)

sys_d3 = ProductionSystem(
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
res_d3 = sys_d3.run(data, verbose=False)
equity_d3 = res_d3.equity_curve
print(f'   (d3) 최종 자본 : ${equity_d3.iloc[-1]:,.0f}')


# ============================================================
# 4. (d3-forced) — (d3) 시스템에 (b)의 페어 풀 강제 고정
# ============================================================
print()
print('=' * 72)
print('4. (d3-forced) (b)의 페어 풀을 강제 고정 + 켈리/필터는 유지')
print('=' * 72)

sys_d3_forced = ProductionSystem(
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
    fixed_pair_pool=fixed_pool,  # ← 9차-A 핵심
)
res_d3f = sys_d3_forced.run(data, verbose=False)
equity_d3f = res_d3f.equity_curve
print(f'   (d3-forced) 최종 자본: ${equity_d3f.iloc[-1]:,.0f}')


# ============================================================
# 5. 분해 A: 거래비용 누적 비교
# ============================================================
print()
print('=' * 72)
print('5. 분해 A — 거래비용 누적 비교')
print('=' * 72)

# (d3)의 페어별 거래비용 추정
# pair_results에서 trades 정보를 다시 꺼냄
d3_total_cost = 0.0
d3_total_trades = 0

# (d3)에서 운용된 페어 풀의 unique set
all_pair_ids_d3 = set()
for st in res_d3.monthly_states:
    for p in st.active_pairs:
        all_pair_ids_d3.add(f'{p.y}~{p.x}')

# 각 페어에 대해 전체 trade 수 계산 — 단, (d3)는 매월 다른 가중치를 줌
# 정확한 cost를 계산하려면: 각 페어의 trades × cost_per_round_trip × 가중치_at_that_time
# 단순화: 평균 가중치 × trade 수
d3_pair_costs = {}
for pid in all_pair_ids_d3:
    # 각 월에서 이 페어의 가중치
    weights = [st.pair_weights.get(pid, 0) for st in res_d3.monthly_states]
    avg_w = np.mean([w for w in weights if w > 0]) if any(w > 0 for w in weights) else 0
    n_months_active = sum(1 for w in weights if w > 0)
    # 해당 페어의 백테스트로 trade 수 추정 (전체 기간 기준)
    p_obj = next((p for st in res_d3.monthly_states
                  for p in st.active_pairs
                  if f'{p.y}~{p.x}' == pid), None)
    if p_obj is None:
        continue
    try:
        spread = compute_spread(data[p_obj.y], data[p_obj.x], p_obj.beta)
        sw = max(20, min(60, int(p_obj.half_life * 2)))
        sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
        z, pos, fc = sg.generate(spread)
        bt = Backtester(initial_capital=INITIAL_CAPITAL,
                        capital_fraction=1.0,
                        fee_rate=FEE, slippage=SLIP)
        res_p = bt.run(data[p_obj.y], data[p_obj.x], p_obj.beta, pos, fc)
        # 운용된 개월 / 전체 백테스트 개월 비율로 trade 비례
        total_months = len(res_d3.monthly_states)
        active_ratio = n_months_active / total_months if total_months > 0 else 0
        effective_trades = len(res_p.trades) * active_ratio
        cost = effective_trades * COST_PER_ROUND_TRIP * avg_w * INITIAL_CAPITAL
        d3_pair_costs[pid] = {
            'n_trades_est': effective_trades,
            'avg_weight': avg_w,
            'cost': cost,
        }
        d3_total_cost += cost
        d3_total_trades += effective_trades
    except Exception:
        pass

# (d3-forced)도 같은 방식
d3f_total_cost = 0.0
d3f_total_trades = 0
for pid in {f'{p.y}~{p.x}' for p in fixed_pool}:
    weights = [st.pair_weights.get(pid, 0) for st in res_d3f.monthly_states]
    avg_w = np.mean([w for w in weights if w > 0]) if any(w > 0 for w in weights) else 0
    n_months_active = sum(1 for w in weights if w > 0)
    p_obj = next((p for p in fixed_pool if f'{p.y}~{p.x}' == pid), None)
    if p_obj is None:
        continue
    try:
        spread = compute_spread(data[p_obj.y], data[p_obj.x], p_obj.beta)
        sw = max(20, min(60, int(p_obj.half_life * 2)))
        sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
        z, pos, fc = sg.generate(spread)
        bt = Backtester(initial_capital=INITIAL_CAPITAL,
                        capital_fraction=1.0,
                        fee_rate=FEE, slippage=SLIP)
        res_p = bt.run(data[p_obj.y], data[p_obj.x], p_obj.beta, pos, fc)
        total_months = len(res_d3f.monthly_states)
        active_ratio = n_months_active / total_months if total_months > 0 else 0
        effective_trades = len(res_p.trades) * active_ratio
        cost = effective_trades * COST_PER_ROUND_TRIP * avg_w * INITIAL_CAPITAL
        d3f_total_cost += cost
        d3f_total_trades += effective_trades
    except Exception:
        pass

print(f'   (b)         총 거래 ≈ {b_total_trades:>5.0f}회, 비용 ≈ ${b_total_cost:>8,.0f}')
print(f'   (d3)        총 거래 ≈ {d3_total_trades:>5.0f}회, 비용 ≈ ${d3_total_cost:>8,.0f}')
print(f'   (d3-forced) 총 거래 ≈ {d3f_total_trades:>5.0f}회, 비용 ≈ ${d3f_total_cost:>8,.0f}')
print()
print(f'   비용 격차 (d3 - b)        : ${d3_total_cost - b_total_cost:>+8,.0f}')
print(f'   비용 격차 (d3-forced - b) : ${d3f_total_cost - b_total_cost:>+8,.0f}')


# ============================================================
# 6. 분해 B: 페어 회전 횟수
# ============================================================
print()
print('=' * 72)
print('6. 분해 B — 페어 회전 횟수')
print('=' * 72)

# (b)는 고정 → 회전 0
# (d3)의 회전 = 매월 새로 들어온 페어 수 합산
n_new_d3 = 0
n_dropped_d3 = 0
pair_durations_d3 = defaultdict(int)  # 페어ID → 활성 개월 수
prev_pids = set()
for st in res_d3.monthly_states:
    cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
    new = cur_pids - prev_pids
    dropped = prev_pids - cur_pids
    n_new_d3 += len(new)
    n_dropped_d3 += len(dropped)
    for pid in cur_pids:
        pair_durations_d3[pid] += 1
    prev_pids = cur_pids

durations_list = list(pair_durations_d3.values())
total_months = len(res_d3.monthly_states)
unique_pairs_d3 = len(pair_durations_d3)

print(f'   (b)  운용 페어     : {len(fixed_pool)}개 (3년 내내 고정)')
print(f'   (b)  회전 횟수     : 0회')
print()
print(f'   (d3) 총 개월       : {total_months}개월')
print(f'   (d3) 유니크 페어 수: {unique_pairs_d3}개')
print(f'   (d3) 신규 진입 횟수: {n_new_d3}회')
print(f'   (d3) 폐기 횟수     : {n_dropped_d3}회')
print(f'   (d3) 페어 평균 수명: {np.mean(durations_list):.1f}개월')
print(f'   (d3) 페어 최대 수명: {max(durations_list)}개월 ({total_months}개월 중)')


# ============================================================
# 7. 분해 C: 신규 페어 진입 후 30일 PnL 분포
# ============================================================
print()
print('=' * 72)
print('7. 분해 C — 신규 페어 진입 후 30일 PnL 분포')
print('=' * 72)

# (d3)의 monthly_states를 훑으며 새 페어 진입 시점 추적
# 각 페어가 처음 활성화된 시점부터 30일간 daily_return 평균
prev_pids = set()
first_active = {}  # 페어ID → 첫 활성 날짜
for st in res_d3.monthly_states:
    cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
    for pid in cur_pids - prev_pids:
        if pid not in first_active:
            first_active[pid] = st.date
    prev_pids = cur_pids

# 각 페어의 첫 30일 PnL 수집
# pair_results를 재구성: ProductionSystem 내부에서 만든 것과 동일하게
# 단순화: 각 페어의 전체 백테스트에서 first_active부터 30일 추출
post_entry_returns = []  # 모든 신규 진입 30일 일별 수익률을 모음
new_pair_summary = []
for pid, entry_date in first_active.items():
    p_obj = next((p for st in res_d3.monthly_states
                  for p in st.active_pairs
                  if f'{p.y}~{p.x}' == pid), None)
    if p_obj is None:
        continue
    try:
        spread = compute_spread(data[p_obj.y], data[p_obj.x], p_obj.beta)
        sw = max(20, min(60, int(p_obj.half_life * 2)))
        sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
        z, pos, fc = sg.generate(spread)
        bt = Backtester(initial_capital=INITIAL_CAPITAL,
                        capital_fraction=1.0,
                        fee_rate=FEE, slippage=SLIP)
        res_p = bt.run(data[p_obj.y], data[p_obj.x], p_obj.beta, pos, fc)
        daily = res_p.equity.pct_change().fillna(0)
        # entry_date 이후 30일
        start_loc = daily.index.searchsorted(entry_date)
        post_30 = daily.iloc[start_loc:start_loc + 30]
        if len(post_30) >= 10:
            avg_ret_30d = post_30.mean()
            sum_ret_30d = post_30.sum()
            post_entry_returns.extend(post_30.tolist())
            new_pair_summary.append({
                'pid': pid,
                'entry': entry_date,
                'first_30d_total': sum_ret_30d,
                'first_30d_avg_daily': avg_ret_30d,
            })
    except Exception:
        pass

# 같은 기간 randomly chosen control: 페어가 풀에 오래 있은 후의 30일 vs 진입 직후 30일
# 단순화 — 전체 베이스라인은 (d3) 전체 평균 일별 수익률
all_daily_d3 = res_d3.daily_returns
overall_mean = all_daily_d3.mean()

if post_entry_returns:
    post_arr = np.array(post_entry_returns)
    print(f'   신규 페어 진입 데이터 포인트: {len(post_arr)}개 (페어 {len(new_pair_summary)}개)')
    print(f'   진입 후 30일 일별 평균 수익률: {post_arr.mean()*10000:+.2f} bp')
    print(f'   진입 후 30일 일별 표준편차    : {post_arr.std()*10000:.2f} bp')
    print(f'   전체 기간 일별 평균 수익률    : {overall_mean*10000:+.2f} bp (비교)')
    print(f'   진입 직후 양수 일수 비율      : {(post_arr > 0).mean():.1%}')
    print(f'   진입 직후 30일 누적 평균      : {np.mean([s["first_30d_total"] for s in new_pair_summary])*100:+.2f}%')

    print()
    print('   ▷ 페어별 진입 후 30일 누적 수익 (Top/Bottom 5):')
    sorted_pairs = sorted(new_pair_summary, key=lambda x: x['first_30d_total'],
                          reverse=True)
    for s in sorted_pairs[:5]:
        print(f'     {s["pid"]:<18s} {s["entry"].date()}: '
              f'{s["first_30d_total"]*100:+6.2f}%')
    print('     ...')
    for s in sorted_pairs[-5:]:
        print(f'     {s["pid"]:<18s} {s["entry"].date()}: '
              f'{s["first_30d_total"]*100:+6.2f}%')


# ============================================================
# 8. 분해 D: (d3-forced) vs (b) 격차 — "롤링 자체의 손실" 분리
# ============================================================
print()
print('=' * 72)
print('8. 분해 D — 같은 페어로 강제 시 잔여 격차')
print('=' * 72)

b_final = equity_b.iloc[-1]
d3_final = equity_d3.iloc[-1]
d3f_final = equity_d3f.iloc[-1]

b_ret = b_final / INITIAL_CAPITAL - 1
d3_ret = d3_final / INITIAL_CAPITAL - 1
d3f_ret = d3f_final / INITIAL_CAPITAL - 1

print()
print(f'   {"방식":<28s} {"최종":>12s} {"수익률":>10s} {"Sharpe":>8s}')
print(f'   {"-"*28} {"-"*12} {"-"*10} {"-"*8}')

def calc_sharpe(eq):
    r = eq.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

print(f'   {"(b) 고정 풀 + 균등":<28s} ${b_final:>10,.0f}  {b_ret:>+8.2%} {calc_sharpe(equity_b):>+8.2f}')
print(f'   {"(d3) 롤링 + 켈리 + 필터":<28s} ${d3_final:>10,.0f}  {d3_ret:>+8.2%} {calc_sharpe(equity_d3):>+8.2f}')
print(f'   {"(d3-forced) 고정 + 켈리 + 필터":<28s} ${d3f_final:>10,.0f}  {d3f_ret:>+8.2%} {calc_sharpe(equity_d3f):>+8.2f}')

# 격차 분해
total_gap = d3_ret - b_ret
rolling_gap = d3_ret - d3f_ret
allocation_gap = d3f_ret - b_ret
print()
print(f'   ▷ 총 격차 (d3 - b)         : {total_gap*100:+.2f}%p')
print(f'   ▷ 롤링 자체 손실 (d3 - d3-forced): {rolling_gap*100:+.2f}%p ← 페어 갈아타기 효과')
print(f'   ▷ 켈리/필터 손실 (d3-forced - b) : {allocation_gap*100:+.2f}%p ← 배분/필터 효과')

if abs(rolling_gap) > abs(allocation_gap):
    main_culprit = '롤링 자체 (페어 갈아타기)'
else:
    main_culprit = '켈리/필터 (자본 배분)'
print()
print(f'   ▷ 주범: {main_culprit}')


# ============================================================
# 9. 시각화
# ============================================================
print()
print('=' * 72)
print('9. 시각화')
print('=' * 72)

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.28)

# (1, full row) Equity 3-way 비교
ax = fig.add_subplot(gs[0, :])
ax.plot(equity_b.index, equity_b, label=f'(b) Fixed pool + Equal  → ${equity_b.iloc[-1]:,.0f}',
        color='steelblue', lw=2.0)
ax.plot(equity_d3.index, equity_d3, label=f'(d3) Rolling + Kelly + Filter  → ${equity_d3.iloc[-1]:,.0f}',
        color='darkgreen', lw=2.0)
ax.plot(equity_d3f.index, equity_d3f, label=f'(d3-forced) Fixed pool + Kelly + Filter  → ${equity_d3f.iloc[-1]:,.0f}',
        color='purple', lw=2.0, ls='--')
ax.axhline(INITIAL_CAPITAL, color='black', ls=':', alpha=0.5)
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title('MVP 9-A: Equity curves — isolating the rolling effect')
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) Cost & trades comparison
ax = fig.add_subplot(gs[1, 0])
labels = ['(b)', '(d3)', '(d3-forced)']
costs = [b_total_cost, d3_total_cost, d3f_total_cost]
trades = [b_total_trades, d3_total_trades, d3f_total_trades]
x = np.arange(len(labels))
ax2 = ax.twinx()
ax.bar(x - 0.2, costs, 0.4, color=['steelblue', 'darkgreen', 'purple'],
       alpha=0.85, label='Cost ($)')
ax2.bar(x + 0.2, trades, 0.4, color=['steelblue', 'darkgreen', 'purple'],
        alpha=0.45, edgecolor='black', label='Trades')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Cumulative trading cost ($)', color='black')
ax2.set_ylabel('Total trades count', color='gray')
ax.set_title('Decomposition A — Cumulative cost & trade count')
ax.grid(alpha=0.3, axis='y')

# (2, 2) Pair turnover (d3 only) — new pairs per month
ax = fig.add_subplot(gs[1, 1])
prev_pids2 = set()
turnover_per_month = []
month_dates = []
for st in res_d3.monthly_states:
    cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
    n_new = len(cur_pids - prev_pids2)
    turnover_per_month.append(n_new)
    month_dates.append(st.date)
    prev_pids2 = cur_pids

ax.bar(month_dates, turnover_per_month, width=pd.Timedelta(days=15),
       color='darkgreen', alpha=0.7)
ax.axhline(np.mean(turnover_per_month), color='red', ls='--',
           label=f'avg {np.mean(turnover_per_month):.1f} new/month')
ax.set_ylabel('New pairs entering pool')
ax.set_xlabel('Date')
ax.set_title(f'Decomposition B — (d3) pair turnover: {n_new_d3} new entries')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3, axis='y')

# (3, 1) Distribution: first 30-day returns for new pairs
ax = fig.add_subplot(gs[2, 0])
if post_entry_returns:
    arr = np.array(post_entry_returns) * 10000  # bp
    ax.hist(arr, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', lw=1.0)
    ax.axvline(arr.mean(), color='red', ls='--',
               label=f'mean {arr.mean():+.1f} bp')
    ax.axvline(overall_mean * 10000, color='blue', ls=':',
               label=f'overall mean {overall_mean*10000:+.1f} bp')
    ax.set_xlabel('Daily return (bp)')
    ax.set_ylabel('Frequency')
    ax.set_title('Decomposition C — First 30-day daily returns for new pairs')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

# (3, 2) Gap decomposition (D)
ax = fig.add_subplot(gs[2, 1])
components = ['Total gap\n(d3 - b)', 'Rolling effect\n(d3 - d3-forced)',
              'Allocation/filter\n(d3-forced - b)']
values = [total_gap * 100, rolling_gap * 100, allocation_gap * 100]
colors_g = ['darkred', 'orange', 'steelblue']
bars = ax.bar(components, values, color=colors_g, alpha=0.85)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Return gap (%p)')
ax.set_title('Decomposition D — Where does the (d3) underperformance come from?')
for bar, v in zip(bars, values):
    if v >= 0:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f'{v:+.2f}%p',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.3, f'{v:+.2f}%p',
                ha='center', va='top', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(min(values) - 2, max(values) + 1.5)

plt.suptitle('MVP 9-A: Decomposing the (b) vs (d3) performance gap',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9a_decomposition.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'   차트 저장: {out_path}')


# ============================================================
# 10. 결론
# ============================================================
print()
print('=' * 72)
print('10. 핵심 결론')
print('=' * 72)
print()

print(f'  격차 분해 결과:')
print(f'    (b) 고정 풀 + 균등        : {b_ret:+.2%}')
print(f'    (d3-forced) 고정 + 켈리   : {d3f_ret:+.2%}')
print(f'    (d3) 롤링 + 켈리 + 필터   : {d3_ret:+.2%}')
print()
print(f'    ↳ 켈리/필터 효과 (b → d3-forced): {allocation_gap*100:+.2f}%p')
print(f'    ↳ 롤링 효과    (d3-forced → d3): {rolling_gap*100:+.2f}%p')
print()

# 결과 dump
import json
report_data = {
    'gap_decomposition': {
        'b_return': b_ret,
        'd3_return': d3_ret,
        'd3_forced_return': d3f_ret,
        'total_gap': total_gap,
        'rolling_gap': rolling_gap,
        'allocation_gap': allocation_gap,
        'main_culprit': main_culprit,
    },
    'cost_analysis': {
        'b_total_cost': b_total_cost,
        'd3_total_cost': d3_total_cost,
        'd3f_total_cost': d3f_total_cost,
        'b_total_trades': b_total_trades,
        'd3_total_trades': d3_total_trades,
        'd3f_total_trades': d3f_total_trades,
    },
    'turnover': {
        'b_n_pairs': len(fixed_pool),
        'd3_unique_pairs': unique_pairs_d3,
        'd3_new_entries': n_new_d3,
        'd3_dropped': n_dropped_d3,
        'd3_avg_duration_months': float(np.mean(durations_list)),
    },
    'new_pair_30d': {
        'n_pairs_observed': len(new_pair_summary),
        'mean_daily_bp': float(np.array(post_entry_returns).mean() * 10000) if post_entry_returns else 0,
        'overall_mean_bp': float(overall_mean * 10000),
        'positive_day_pct': float((np.array(post_entry_returns) > 0).mean()) if post_entry_returns else 0,
        'mean_30d_total_pct': float(np.mean([s['first_30d_total'] for s in new_pair_summary]) * 100) if new_pair_summary else 0,
    },
}
data_dump_path = '/tmp/mvp9a_report_data.json'
with open(data_dump_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print(f'  보고서용 데이터 저장: {data_dump_path}')
