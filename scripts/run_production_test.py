"""
7차 MVP: 통합 시스템 실전 시뮬레이션
=====================================

비교 대상:
  - 단일 페어 운용 (1차 MVP 방식): GDX~KO 한 페어
  - 다중 페어 + 고정 풀 (2차 MVP 방식): 7개 페어 고정
  - 다중 페어 + 롤링 + 균등 배분 (6차 MVP 확장)
  - 다중 페어 + 롤링 + 포트폴리오 켈리 (7차 MVP — 완전 통합)

기대:
  - 시간이 갈수록 (단일 → 고정 → 롤링 → 포트폴리오) 안정성과 수익이 개선
  - 6차 MVP의 페어 풀 회전을 자본 배분에 반영
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, Pair, SignalGenerator, Backtester, compute_spread,
)
from rolling_pairs import RollingPairsManager
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

print('=' * 70)
print('1. 데이터 다운로드')
print('=' * 70)
data = yf.download(TICKERS, period='3y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
print(f'  {data.shape[0]}일 x {data.shape[1]}종목')

INITIAL_CAPITAL = 100_000


# ============================================================
# 2. 4-way 비교 백테스트
# ============================================================
print()
print('=' * 70)
print('2. 4가지 운용 방식 비교')
print('=' * 70)

results = {}

# (a) 단일 페어 (GDX~KO만)
print()
print('  (a) 단일 페어 (GDX~KO만)')
finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)
gdx_ko_beta = finder.tls_beta(
    np.log(data['KO'].iloc[:int(len(data)*0.7)]).values,
    np.log(data['GDX'].iloc[:int(len(data)*0.7)]).values)
sp = np.log(data['GDX']) - gdx_ko_beta * np.log(data['KO'])
hl_train = finder.half_life(sp.iloc[:int(len(data)*0.7)]) or 14.7
gdx_ko_pair = Pair(y='GDX', x='KO', beta=gdx_ko_beta, pvalue=0.05,
                   half_life=hl_train)
sg = SignalGenerator(window=29, entry=2.0, exit_thr=0.5, stop=3.5)
spread_full = compute_spread(data['GDX'], data['KO'], gdx_ko_beta)
z, pos, fc = sg.generate(spread_full)
bt = Backtester(initial_capital=INITIAL_CAPITAL,
               capital_fraction=0.10,
               fee_rate=0.0004, slippage=0.0005)
res_single = bt.run(data['GDX'], data['KO'], gdx_ko_beta, pos, fc)
results['(a) Single GDX~KO'] = res_single.equity
print(f'      최종 자본: ${res_single.equity.iloc[-1]:,.0f}, '
      f'Sharpe {res_single.metrics["sharpe"]:.2f}')

# (b) 고정 페어 풀 (1차 MVP가 발굴한 7개 페어로 고정 운용)
print('  (b) 고정 페어 풀 — 1차 MVP의 7개 페어 균등 배분')
fixed_pairs_list = []
# 1차 MVP 페어들 reconstructing (페어 풀 생성)
all_pairs = finder.screen_pairs(data.iloc[:int(len(data)*0.7)])
print(f'      Train에서 발굴: {len(all_pairs)}개 페어')

if all_pairs:
    n = len(all_pairs)
    equal_w = 0.10 / n  # 총 자본의 10%를 균등 배분
    daily_returns_fixed = pd.Series(0.0, index=data.index)

    for p in all_pairs:
        try:
            spread = compute_spread(data[p.y], data[p.x], p.beta)
            sw = max(20, min(60, int(p.half_life * 2)))
            sgg = SignalGenerator(window=sw, entry=2.0,
                                  exit_thr=0.5, stop=3.5)
            zz, pp, ff = sgg.generate(spread)
            bb = Backtester(initial_capital=INITIAL_CAPITAL,
                          capital_fraction=1.0,
                          fee_rate=0.0004, slippage=0.0005)
            rr = bb.run(data[p.y], data[p.x], p.beta, pp, ff)
            r_daily = rr.equity.pct_change().fillna(0)
            daily_returns_fixed += equal_w * r_daily
        except Exception:
            pass

    equity_fixed = INITIAL_CAPITAL * (1 + daily_returns_fixed).cumprod()
    results['(b) Fixed pool (MVP1 pairs)'] = equity_fixed
    print(f'      최종 자본: ${equity_fixed.iloc[-1]:,.0f}')

# (c) 롤링 페어 풀 + 균등 배분
print('  (c) 롤링 풀 + 균등 배분')
sys_equal = ProductionSystem(
    finder=finder,
    lookback_days=365,
    refresh_every_days=30,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=INITIAL_CAPITAL,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=False,  # 균등 배분
)
res_rolling_eq = sys_equal.run(data, verbose=False)
results['(c) Rolling pool + Equal-weight'] = res_rolling_eq.equity_curve
print(f'      최종 자본: ${res_rolling_eq.equity_curve.iloc[-1]:,.0f}, '
      f'Sharpe {res_rolling_eq.metrics["sharpe"]:.2f}')

# (d) 롤링 페어 풀 + 포트폴리오 켈리 (완전 통합)
print('  (d) 롤링 풀 + 포트폴리오 켈리 (7차 MVP 완전판)')
sys_kelly = ProductionSystem(
    finder=finder,
    lookback_days=365,
    refresh_every_days=30,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=INITIAL_CAPITAL,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=True,  # 켈리 사용
)
res_full = sys_kelly.run(data, verbose=True)
results['(d) Full system (rolling + Kelly)'] = res_full.equity_curve
print(f'\n      최종 자본: ${res_full.equity_curve.iloc[-1]:,.0f}, '
      f'Sharpe {res_full.metrics["sharpe"]:.2f}')


# ============================================================
# 3. 종합 비교
# ============================================================
print()
print('=' * 70)
print('3. 4가지 운용 방식 종합 비교')
print('=' * 70)
print()
print(f'  {"방식":<40s} {"최종 자본":>14s} {"수익률":>10s} {"Sharpe":>8s} {"MDD":>8s}')

# (a) 단일
m_a = res_single.metrics
print(f'  {"(a) 단일 GDX~KO":<40s} '
      f'${res_single.equity.iloc[-1]:>12,.0f}   '
      f'{m_a["total_return"]:>+9.2%} '
      f'{m_a["sharpe"]:>+8.2f} '
      f'{m_a["max_drawdown"]:>+8.2%}')

# (b) 고정
if '(b) Fixed pool (MVP1 pairs)' in results:
    eq_b = results['(b) Fixed pool (MVP1 pairs)']
    ret_b = eq_b.iloc[-1] / eq_b.iloc[0] - 1
    daily_b = eq_b.pct_change().dropna()
    sh_b = daily_b.mean() / daily_b.std() * np.sqrt(252) if daily_b.std() > 0 else 0
    mdd_b = (eq_b / eq_b.cummax() - 1).min()
    print(f'  {"(b) 고정 풀 (1차 MVP 7개)":<40s} '
          f'${eq_b.iloc[-1]:>12,.0f}   '
          f'{ret_b:>+9.2%} '
          f'{sh_b:>+8.2f} '
          f'{mdd_b:>+8.2%}')

# (c) 롤링 균등
m_c = res_rolling_eq.metrics
print(f'  {"(c) 롤링 풀 + 균등 배분":<40s} '
      f'${res_rolling_eq.equity_curve.iloc[-1]:>12,.0f}   '
      f'{m_c["total_return"]:>+9.2%} '
      f'{m_c["sharpe"]:>+8.2f} '
      f'{m_c["max_drawdown"]:>+8.2%}')

# (d) 완전 통합
m_d = res_full.metrics
print(f'  {"(d) 완전 통합 (롤링+켈리)":<40s} '
      f'${res_full.equity_curve.iloc[-1]:>12,.0f}   '
      f'{m_d["total_return"]:>+9.2%} '
      f'{m_d["sharpe"]:>+8.2f} '
      f'{m_d["max_drawdown"]:>+8.2%}')


# ============================================================
# 4. 페어별 누적 PnL (어떤 페어가 진짜 돈을 벌었나)
# ============================================================
print()
print('=' * 70)
print('4. 페어별 누적 PnL — 완전 통합 시스템 (d)')
print('=' * 70)
print()
sorted_pnl = sorted(res_full.pair_lifetime_pnl.items(),
                    key=lambda x: x[1], reverse=True)
print('  Top 10 수익 페어:')
for i, (pid, pnl) in enumerate(sorted_pnl[:10], 1):
    print(f'    {i:>2d}. {pid:<18s} ${pnl:>+10,.0f}')

print('\n  Bottom 5 손실 페어:')
for i, (pid, pnl) in enumerate(sorted_pnl[-5:], 1):
    print(f'    {i:>2d}. {pid:<18s} ${pnl:>+10,.0f}')


# ============================================================
# 5. 시각화
# ============================================================
print()
print('=' * 70)
print('5. 시각화')
print('=' * 70)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)

# (1, full row) 자본 곡선 4-way 비교
ax = fig.add_subplot(gs[0, :])
colors = {
    '(a) Single GDX~KO': 'gray',
    '(b) Fixed pool (MVP1 pairs)': 'steelblue',
    '(c) Rolling pool + Equal-weight': 'darkorange',
    '(d) Full system (rolling + Kelly)': 'darkgreen',
}
for name, eq in results.items():
    ax.plot(eq.index, eq, label=name, color=colors.get(name, 'black'),
            lw=1.8 if 'Full' in name else 1.3,
            alpha=1.0 if 'Full' in name else 0.7)
ax.axhline(INITIAL_CAPITAL, color='red', ls=':', alpha=0.5,
           label=f'Initial capital ${INITIAL_CAPITAL:,}')
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title('Equity curves — 4 trading approaches')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) 매월 운용 페어 수
ax = fig.add_subplot(gs[1, 0])
monthly_dates = [s.date for s in res_full.monthly_states]
n_active = [len(s.active_pairs) for s in res_full.monthly_states]
ax.bar(monthly_dates, n_active, width=pd.Timedelta(days=20),
       color='darkgreen', alpha=0.7)
ax.set_ylabel('Active pairs per month')
ax.set_xlabel('Date')
ax.set_title('How many pairs were active each month?')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3, axis='y')

# (2, 2) 페어별 누적 PnL (top 10 + bottom 5)
ax = fig.add_subplot(gs[1, 1])
display = sorted_pnl[:10] + sorted_pnl[-5:]
display_ids = [d[0] for d in display]
display_pnl = [d[1] for d in display]
display_colors = ['darkgreen' if p > 0 else 'red' for p in display_pnl]
ax.barh(range(len(display)), display_pnl, color=display_colors, alpha=0.7)
ax.set_yticks(range(len(display)))
ax.set_yticklabels(display_ids, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('Cumulative PnL ($)')
ax.set_title('Per-pair lifetime PnL (top 10 + bottom 5)')
ax.grid(alpha=0.3, axis='x')

# (3, 1) Drawdown 비교
ax = fig.add_subplot(gs[2, 0])
for name, eq in results.items():
    dd = (eq / eq.cummax() - 1) * 100
    ax.plot(dd.index, dd, label=name, color=colors.get(name, 'black'),
            lw=1.5 if 'Full' in name else 1.0,
            alpha=1.0 if 'Full' in name else 0.7)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.set_title('Drawdown comparison — who loses less in bad times?')
ax.legend(loc='lower left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (3, 2) 최종 지표 막대 비교
ax = fig.add_subplot(gs[2, 1])
labels = []
sharpes = []
returns = []
for name in ['(a) Single GDX~KO',
              '(b) Fixed pool (MVP1 pairs)',
              '(c) Rolling pool + Equal-weight',
              '(d) Full system (rolling + Kelly)']:
    if name not in results:
        continue
    eq = results[name]
    daily = eq.pct_change().dropna()
    sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    short_name = name.split(') ')[0] + ')'
    labels.append(short_name)
    sharpes.append(sh)
    returns.append(ret * 100)  # %

x = np.arange(len(labels))
ax2 = ax.twinx()
bars1 = ax.bar(x - 0.2, sharpes, 0.4, color='steelblue',
              alpha=0.8, label='Sharpe (left)')
bars2 = ax2.bar(x + 0.2, returns, 0.4, color='darkorange',
               alpha=0.8, label='Return % (right)')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Sharpe ratio', color='steelblue')
ax2.set_ylabel('Total return (%)', color='darkorange')
ax.set_title('Final metrics — Sharpe vs Total Return')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('7차 MVP: 1~6차 발견 모두 통합 — 4-way 비교',
             fontsize=13, y=0.995)

out_path = '/home/claude/production_comparison_chart.png'
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

final_a = res_single.equity.iloc[-1]
final_b = results['(b) Fixed pool (MVP1 pairs)'].iloc[-1] \
          if '(b) Fixed pool (MVP1 pairs)' in results else INITIAL_CAPITAL
final_c = res_rolling_eq.equity_curve.iloc[-1]
final_d = res_full.equity_curve.iloc[-1]

print(f'  초기 자본 $100,000 → 3년 후:')
print(f'    (a) 단일 페어        : ${final_a:>10,.0f}  ({(final_a/INITIAL_CAPITAL-1)*100:+.2f}%)')
print(f'    (b) 고정 풀 (7개)    : ${final_b:>10,.0f}  ({(final_b/INITIAL_CAPITAL-1)*100:+.2f}%)')
print(f'    (c) 롤링 + 균등      : ${final_c:>10,.0f}  ({(final_c/INITIAL_CAPITAL-1)*100:+.2f}%)')
print(f'    (d) 완전 통합        : ${final_d:>10,.0f}  ({(final_d/INITIAL_CAPITAL-1)*100:+.2f}%)')
print()
print(f'  Sharpe 비교:')
print(f'    (a) 단일             : {res_single.metrics["sharpe"]:+.2f}')
print(f'    (c) 롤링 + 균등      : {res_rolling_eq.metrics["sharpe"]:+.2f}')
print(f'    (d) 완전 통합        : {res_full.metrics["sharpe"]:+.2f}')
