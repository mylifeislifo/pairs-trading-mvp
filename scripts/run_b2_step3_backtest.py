"""
B-2 STEP 3: 델타 뉴트럴 펀딩비 차익 백테스트 (일봉 단위)

데이터:
  - 가격: OKX 일봉 (spot + perp), 2020-01 ~ 2026-05 (2333일)
  - Funding: Hyperliquid 시간당 → 일 단위 합산 (2023-05 ~ 2026-05)
  - OKX funding은 90일치만 — 사용 안 함

기간: 2023-05-12 ~ 2026-05-21 (Hyperliquid funding이 시작되는 시점부터)

포지션:
  - notional $100K total
  - $50K spot long + $50K perp short (델타 뉴트럴)
  - 진입 1번, 청산 1번 (단순 buy-and-hold 펀딩 수취)

손익 (일별):
  spot pnl     = $50K × (spot_today - spot_yest) / spot_yest
  perp pnl     = $50K × (perp_yest - perp_today) / perp_yest  # short
  funding pnl  = $50K × daily_funding_sum  # short이 펀딩 받음 (양수일 때)
  
  total = spot + perp + funding - (진입일/마지막일 거래비용)

거래비용:
  - Spot taker: 0.08% (보수적)
  - Perp taker: 0.04% (perp 일반)
  - 한 사이클 진입+청산 = (0.08 + 0.04) × 2 = 0.24% (현실적 보수)

검증:
  1. 전체 기간 손익
  2. 연도별 분해 (시기 의존성)
  3. Walk-forward (Train 6m + Test 3m + Step 1m)
     - Train에서 펀딩비 양수 비율 확인 후 Test에서 운용
  4. (b) 비교: 단순 BTC long-only buy-and-hold (방향성 비교용)
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))


# ============================================================
# 데이터 정리
# ============================================================
def prep_data(coin: str):
    """coin → spot price, perp price, daily funding (sum of hourly)"""
    spot = cache[f'{coin}_spot_1d'][['close']].rename(columns={'close': 'spot'})
    perp = cache[f'{coin}_perp_1d'][['close']].rename(columns={'close': 'perp'})
    fund_hourly = cache[f'HL_{coin}_funding'][['fundingRate']]

    # 모든 인덱스를 date (00:00 normalize)로 통일
    spot.index = spot.index.normalize()
    perp.index = perp.index.normalize()

    # Hyperliquid funding을 일 단위로 합산
    fund_hourly = fund_hourly.copy()
    fund_hourly['date'] = fund_hourly.index.normalize()
    fund_daily = fund_hourly.groupby('date')['fundingRate'].sum()
    fund_daily.name = 'funding_daily'
    fund_daily.index.name = None

    # 공통 인덱스로 정렬
    df = spot.join(perp, how='inner').join(fund_daily, how='inner')
    df = df.dropna()
    df['spot_ret'] = df['spot'].pct_change()
    df['perp_ret'] = df['perp'].pct_change()
    df = df.dropna()
    return df


btc = prep_data('BTC')
eth = prep_data('ETH')

print(f'BTC: {len(btc)}일, {btc.index[0].date()} ~ {btc.index[-1].date()}')
print(f'ETH: {len(eth)}일')
print()

# 펀딩비 일일 통계
for coin, df in [('BTC', btc), ('ETH', eth)]:
    fd = df['funding_daily']
    annual = fd.mean() * 365
    print(f'{coin} daily funding: mean {fd.mean()*100:+.5f}%, annual {annual*100:+.2f}%, pos% {(fd > 0).mean()*100:.1f}%')
print()


# ============================================================
# 백테스트 엔진
# ============================================================
def backtest_delta_neutral(df: pd.DataFrame,
                           initial_capital: float = 100_000,
                           spot_fee: float = 0.0008,   # 0.08%
                           perp_fee: float = 0.0004,   # 0.04%
                           ) -> pd.DataFrame:
    """
    델타 뉴트럴 펀딩 차익 백테스트.
    
    포지션:
      $50K spot long + $50K perp short = $100K notional, $100K 자본 사용
      (실전에선 선물 마진 일부 + 현물 일부지만 단순화)
    
    매일:
      pnl = spot_ret × 50K - perp_ret × 50K + funding_daily × 50K
            (펀딩 양수면 short이 받음)
    
    거래비용: 진입일 (spot_fee + perp_fee) × notional, 마지막날 동일
    """
    notional_each = initial_capital / 2

    spot_pnl = df['spot_ret'] * notional_each
    perp_pnl = -df['perp_ret'] * notional_each   # short
    funding_pnl = df['funding_daily'] * notional_each  # short receives

    daily_pnl = spot_pnl + perp_pnl + funding_pnl

    # 거래비용: 진입 (첫날) + 청산 (마지막날)
    entry_cost = (spot_fee + perp_fee) * notional_each * 2  # 양다리
    daily_pnl.iloc[0] -= entry_cost
    # exit cost는 청산 시 — 단순화: 마지막 날에 차감
    exit_cost = (spot_fee + perp_fee) * notional_each * 2
    daily_pnl.iloc[-1] -= exit_cost

    equity = initial_capital + daily_pnl.cumsum()
    return pd.DataFrame({
        'equity': equity,
        'daily_pnl': daily_pnl,
        'spot_pnl': spot_pnl,
        'perp_pnl': perp_pnl,
        'funding_pnl': funding_pnl,
        'tracking_error': spot_pnl + perp_pnl,  # 베이시스 추적 오차
    })


# ============================================================
# 전체 기간 백테스트
# ============================================================
print('=' * 72)
print('전체 기간 백테스트')
print('=' * 72)

results = {}
for coin, df in [('BTC', btc), ('ETH', eth)]:
    res = backtest_delta_neutral(df)
    days = len(res)
    total_ret = res['equity'].iloc[-1] / 100_000 - 1
    annual = (1 + total_ret) ** (365 / days) - 1
    daily = res['equity'].pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    mdd = (res['equity'] / res['equity'].cummax() - 1).min()
    
    # PnL 분해
    total_funding = res['funding_pnl'].sum()
    total_tracking = res['tracking_error'].sum()
    total_cost = -2 * (0.0008 + 0.0004) * 50_000 * 2  # entry + exit, both sides

    results[coin] = {
        'res': res, 'df': df,
        'days': days, 'total_ret': total_ret, 'annual': annual,
        'sharpe': sharpe, 'mdd': mdd,
        'funding': total_funding, 'tracking': total_tracking, 'cost': total_cost,
    }

    print(f'\n{coin}:')
    print(f'  기간            : {days}일')
    print(f'  최종 자본       : ${res["equity"].iloc[-1]:,.0f}')
    print(f'  누적 수익률     : {total_ret*100:+.2f}%')
    print(f'  연환산          : {annual*100:+.2f}%')
    print(f'  Sharpe (annual) : {sharpe:+.2f}')
    print(f'  MDD             : {mdd*100:+.2f}%')
    print()
    print(f'  PnL 분해:')
    print(f'    펀딩 수입     : ${total_funding:>+10,.0f}')
    print(f'    추적 오차     : ${total_tracking:>+10,.0f}  (spot - perp 가격 변동)')
    print(f'    거래비용      : ${total_cost:>+10,.0f}')
    print(f'    합계          : ${total_funding + total_tracking + total_cost:>+10,.0f}')


# ============================================================
# 연도별 분해 — 시기 의존성 점검
# ============================================================
print()
print('=' * 72)
print('연도별 분해 — 시기 의존성')
print('=' * 72)
print()
print(f'  {"코인":<5s} {"연도":<7s} {"일수":>4s} {"수익률":>9s} {"연환산":>9s} {"Sharpe":>8s} {"펀딩":>10s} {"추적":>10s}')
print(f'  {"-"*5} {"-"*7} {"-"*4} {"-"*9} {"-"*9} {"-"*8} {"-"*10} {"-"*10}')

annual_results = {}
for coin in ['BTC', 'ETH']:
    df = results[coin]['df']
    res = results[coin]['res']
    annual_results[coin] = []
    for year in [2023, 2024, 2025, 2026]:
        mask = res.index.year == year
        if mask.sum() < 10:
            continue
        sub = res[mask]
        days = len(sub)
        # 연도 시작 자본 → 끝 자본 비율
        sub_eq = sub['equity']
        sub_ret = sub_eq.iloc[-1] / sub_eq.iloc[0] - 1
        annual_rate = (1 + sub_ret) ** (365 / days) - 1 if days > 0 else 0
        daily = sub_eq.pct_change().dropna()
        sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
        funding = sub['funding_pnl'].sum()
        tracking = sub['tracking_error'].sum()
        annual_results[coin].append({
            'year': year, 'days': days, 'ret': sub_ret,
            'annual': annual_rate, 'sharpe': sharpe,
            'funding': funding, 'tracking': tracking,
        })
        print(f'  {coin:<5s} {year:<7d} {days:>4d} '
              f'{sub_ret*100:>+8.2f}% '
              f'{annual_rate*100:>+8.2f}% '
              f'{sharpe:>+7.2f} '
              f'${funding:>+9,.0f} '
              f'${tracking:>+9,.0f}')


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full) Equity curves
ax = fig.add_subplot(gs[0, :])
colors = {'BTC': 'darkorange', 'ETH': 'steelblue'}
for coin in ['BTC', 'ETH']:
    res = results[coin]['res']
    ax.plot(res.index, res['equity'], 
            label=f'{coin} delta-neutral → {results[coin]["total_ret"]*100:+.2f}% '
                  f'({results[coin]["annual"]*100:+.2f}% annual, '
                  f'Sh {results[coin]["sharpe"]:+.2f})',
            color=colors[coin], lw=2)
ax.axhline(100_000, color='black', ls=':', alpha=0.5, label='Initial $100K')
ax.set_ylabel('Equity (USD)')
ax.set_title('Delta-neutral funding arbitrage — equity curves')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) PnL 분해 (BTC)
ax = fig.add_subplot(gs[1, 0])
btc_res = results['BTC']['res']
cum_funding = btc_res['funding_pnl'].cumsum()
cum_tracking = btc_res['tracking_error'].cumsum()
ax.plot(btc_res.index, cum_funding, label='Funding income (cumulative)', color='darkgreen', lw=2)
ax.plot(btc_res.index, cum_tracking, label='Tracking error (spot-perp)', color='red', lw=1.5, alpha=0.7)
ax.plot(btc_res.index, cum_funding + cum_tracking,
        label='Sum (before fees)', color='black', lw=2, ls='--')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Cumulative PnL (USD)')
ax.set_title('BTC: Funding income vs Tracking error decomposition')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 2) 일별 펀딩비 시간 추이
ax = fig.add_subplot(gs[1, 1])
ax.plot(btc.index, btc['funding_daily']*100, color='darkorange', alpha=0.5,
        label='BTC daily funding %', lw=0.8)
# 30일 이동 평균
ma = btc['funding_daily'].rolling(30).mean() * 100
ax.plot(btc.index, ma, color='red', lw=2, label='30-day MA')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Daily funding rate (%)')
ax.set_title('BTC daily funding rate over time')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (3, 1) 연도별 수익률 비교
ax = fig.add_subplot(gs[2, 0])
years = sorted(set(r['year'] for r in annual_results['BTC']))
x = np.arange(len(years))
w = 0.35
btc_annuals = [next((r['annual']*100 for r in annual_results['BTC'] if r['year']==y), 0) for y in years]
eth_annuals = [next((r['annual']*100 for r in annual_results['ETH'] if r['year']==y), 0) for y in years]
ax.bar(x - w/2, btc_annuals, w, color='darkorange', alpha=0.85, label='BTC')
ax.bar(x + w/2, eth_annuals, w, color='steelblue', alpha=0.85, label='ETH')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylabel('Annualized return (%)')
ax.set_title('Annualized return by year')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (3, 2) Drawdown
ax = fig.add_subplot(gs[2, 1])
for coin in ['BTC', 'ETH']:
    res = results[coin]['res']
    dd = (res['equity'] / res['equity'].cummax() - 1) * 100
    ax.plot(dd.index, dd, color=colors[coin], lw=1.5, label=coin)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown over time')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

plt.suptitle('B-2: Delta-neutral funding arbitrage backtest (daily, OKX prices + HL funding)',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'b2_funding_arbitrage.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 결론
# ============================================================
print()
print('=' * 72)
print('결론')
print('=' * 72)
print()

# 거래비용 회수 가능성
for coin in ['BTC', 'ETH']:
    r = results[coin]
    funding_per_year = r['funding'] / r['days'] * 365
    cost_one_time = -r['cost']
    days_to_recover = cost_one_time / (r['funding'] / r['days'])
    print(f'  {coin}:')
    print(f'    연 펀딩 수입 : ${funding_per_year:,.0f}')
    print(f'    거래비용 1회 : ${cost_one_time:,.0f}')
    print(f'    회수 일수    : {days_to_recover:.1f}일')

print()
# 전체 Sharpe + 결론
print('  Walk-forward 검증 다음 단계 추천 — Train 6mo + Test 3mo + Step 1mo')

import json
summary = {
    'period': {'start': str(btc.index[0].date()), 'end': str(btc.index[-1].date()),
               'days': len(btc)},
    'results': {
        coin: {
            'total_return': float(r['total_ret']),
            'annual': float(r['annual']),
            'sharpe': float(r['sharpe']),
            'mdd': float(r['mdd']),
            'funding_pnl': float(r['funding']),
            'tracking_error': float(r['tracking']),
            'total_cost': float(r['cost']),
        }
        for coin, r in results.items()
    },
    'annual_breakdown': annual_results,
}
with open('/tmp/mvp_b2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print('Summary 저장: /tmp/mvp_b2_summary.json')
