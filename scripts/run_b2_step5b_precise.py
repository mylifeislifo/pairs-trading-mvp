"""
B-2 STEP 5b: 1h 정밀 시뮬 vs 일별 모델 비교

7개월 (2025-10-28 ~ 2026-05-22) 동일 데이터로:
1. 1h 정밀 시뮬: notional 시점별, mark price funding
2. 일별 시뮬: notional 고정, 합산 funding
→ PnL 차이 → 일별 모델 편향 정량화

핵심 식 차이:
  일별: funding_pnl = N_init × Σ_t funding_t
  정밀: funding_pnl = Σ_t (N_init/perp_0 × perp_t) × funding_t
                    = (N_init/perp_0) × Σ_t (perp_t × funding_t)

  차이 = (N_init/perp_0) × Σ_t (perp_t - perp_0) × funding_t

  즉, "가격 편차 × funding"의 시점별 합. 가격이 오르고 funding이 양수면 양의 편향.
"""
import os, pickle, json
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
def prep_1h(coin: str):
    """1h spot + 1h perp + 1h funding 정렬"""
    spot = cache[f'OKX_{coin}_spot_1h'][['close']].rename(columns={'close': 'spot'})
    perp = cache[f'OKX_{coin}_perp_1h'][['close']].rename(columns={'close': 'perp'})
    fund = cache[f'HL_{coin}_funding'][['fundingRate']]
    # HL funding은 미세 ms 단위 timestamp → 시간 단위로 정렬
    fund = fund.copy()
    fund.index = fund.index.floor('h')

    df = spot.join(perp, how='inner').join(fund, how='left')
    df['fundingRate'] = df['fundingRate'].fillna(0)
    df = df.dropna(subset=['spot', 'perp']).sort_index()
    return df


btc_1h = prep_1h('BTC')
eth_1h = prep_1h('ETH')
print(f'BTC 1h: {len(btc_1h)}, {btc_1h.index[0]} ~ {btc_1h.index[-1]}')
print(f'  funding 비0 시점: {(btc_1h["fundingRate"] != 0).sum()}')
print(f'ETH 1h: {len(eth_1h)}')
print(f'  funding 비0 시점: {(eth_1h["fundingRate"] != 0).sum()}')


# ============================================================
# 1h 정밀 시뮬
# ============================================================
INITIAL = 100_000
SPOT_FEE = 0.0008
PERP_FEE = 0.0004


def precise_1h_sim(df: pd.DataFrame, capital: float = INITIAL):
    """
    정밀 시뮬:
      진입 t=0:
        spot_qty = (capital/2) / spot[0]  코인 수
        perp_qty = -(capital/2) / perp[0]  short, 음수
      
      매 시점:
        mark_value(t) = spot_qty × spot[t] + perp_qty × perp[t]
                      = (capital/2) × (spot_t/spot_0 - perp_t/perp_0) + capital/2 - capital/2
                      
        그냥 직접:
        mark_pnl(t) = spot_qty×(spot_t - spot_0) + perp_qty×(perp_t - perp_0)
                    = (capital/2)×(spot_t-spot_0)/spot_0 + (-capital/2)×(perp_t-perp_0)/perp_0
      
      Funding payment @ funding time t:
        payment = -perp_qty × perp[t] × funding_rate(t)
                = (capital/2 / perp_0) × perp[t] × funding_rate(t)
    """
    d = df.copy()
    notional_each = capital / 2
    spot0 = d['spot'].iloc[0]
    perp0 = d['perp'].iloc[0]
    spot_qty = notional_each / spot0
    perp_qty = -notional_each / perp0

    d['mark_pnl_cum'] = spot_qty * (d['spot'] - spot0) + perp_qty * (d['perp'] - perp0)
    d['funding_payment'] = -perp_qty * d['perp'] * d['fundingRate']  # 양수 funding → +
    d['cum_funding'] = d['funding_payment'].cumsum()

    # 거래비용: 진입 + 청산
    entry_cost = (SPOT_FEE + PERP_FEE) * notional_each * 2  # 양다리
    exit_cost = (SPOT_FEE + PERP_FEE) * notional_each * 2

    d['equity'] = capital + d['mark_pnl_cum'] + d['cum_funding']
    d.iloc[0, d.columns.get_loc('equity')] -= entry_cost
    d.iloc[-1, d.columns.get_loc('equity')] -= exit_cost
    return d


# ============================================================
# 같은 기간 일별 시뮬 (step 3 모델)
# ============================================================
def prep_daily_from_1h(df_1h: pd.DataFrame):
    """1h 데이터를 일별로 다운샘플 → 일별 모델 입력"""
    daily = pd.DataFrame()
    daily['spot'] = df_1h['spot'].resample('D').last()
    daily['perp'] = df_1h['perp'].resample('D').last()
    daily['funding_daily'] = df_1h['fundingRate'].resample('D').sum()
    daily = daily.dropna()
    daily['spot_ret'] = daily['spot'].pct_change()
    daily['perp_ret'] = daily['perp'].pct_change()
    return daily.dropna()


def daily_sim(d_daily: pd.DataFrame, capital: float = INITIAL):
    """일별 모델 (step 3와 동일)"""
    notional = capital / 2
    spot_pnl = d_daily['spot_ret'] * notional
    perp_pnl = -d_daily['perp_ret'] * notional
    funding_pnl = d_daily['funding_daily'] * notional
    daily = spot_pnl + perp_pnl + funding_pnl

    cost = (SPOT_FEE + PERP_FEE) * notional * 2
    daily.iloc[0] -= cost
    daily.iloc[-1] -= cost

    eq = capital + daily.cumsum()
    return pd.DataFrame({
        'equity': eq,
        'spot_pnl': spot_pnl,
        'perp_pnl': perp_pnl,
        'funding_pnl': funding_pnl,
    })


# ============================================================
# 비교 실행
# ============================================================
print()
print('=' * 72)
print('1h 정밀 vs 일별 모델 비교')
print('=' * 72)

results = {}
for coin, df_1h in [('BTC', btc_1h), ('ETH', eth_1h)]:
    precise = precise_1h_sim(df_1h)
    daily_input = prep_daily_from_1h(df_1h)
    daily_res = daily_sim(daily_input)

    # 일별 최종으로 다운샘플 (비교용)
    precise_eod = precise['equity'].resample('D').last().dropna()

    # 같은 마지막 일자에서 비교
    common = precise_eod.index.intersection(daily_res['equity'].index)
    precise_eod_aligned = precise_eod.loc[common]
    daily_eq_aligned = daily_res['equity'].loc[common]

    # 최종 결과
    p_final = precise['equity'].iloc[-1]
    d_final = daily_res['equity'].iloc[-1]

    p_ret = p_final / INITIAL - 1
    d_ret = d_final / INITIAL - 1
    gap_abs = p_final - d_final
    gap_pct = (p_final - d_final) / INITIAL * 100

    p_funding_total = precise['cum_funding'].iloc[-1]
    d_funding_total = daily_res['funding_pnl'].sum()

    p_tracking_total = precise['mark_pnl_cum'].iloc[-1]
    d_tracking_total = (daily_res['spot_pnl'] + daily_res['perp_pnl']).sum()

    # 일별 차이 (정렬 시점에서)
    daily_diffs = precise_eod_aligned - daily_eq_aligned

    results[coin] = {
        'precise_final': float(p_final),
        'daily_final': float(d_final),
        'precise_return': float(p_ret),
        'daily_return': float(d_ret),
        'gap_abs': float(gap_abs),
        'gap_pct': float(gap_pct),
        'precise_funding': float(p_funding_total),
        'daily_funding': float(d_funding_total),
        'funding_gap': float(p_funding_total - d_funding_total),
        'precise_tracking': float(p_tracking_total),
        'daily_tracking': float(d_tracking_total),
        'max_daily_diff': float(daily_diffs.abs().max()),
        'mean_daily_diff': float(daily_diffs.abs().mean()),
        'precise_obj': precise,
        'daily_obj': daily_res,
    }

    print(f'\n{coin} ({len(df_1h)}시간 = {len(df_1h)//24}일):')
    print(f'  ─── 최종 자본 ───')
    print(f'    정밀 1h : ${p_final:,.0f}  ({p_ret*100:+.2f}%)')
    print(f'    일별    : ${d_final:,.0f}  ({d_ret*100:+.2f}%)')
    print(f'    차이    : ${gap_abs:+,.0f} ({gap_pct:+.3f}%p of initial)')
    print(f'  ─── Funding PnL ───')
    print(f'    정밀    : ${p_funding_total:+,.0f}')
    print(f'    일별    : ${d_funding_total:+,.0f}')
    print(f'    차이    : ${p_funding_total - d_funding_total:+,.0f}  ← notional 효과')
    print(f'  ─── 추적 오차 (spot-perp) ───')
    print(f'    정밀    : ${p_tracking_total:+,.0f}')
    print(f'    일별    : ${d_tracking_total:+,.0f}')
    print(f'  ─── 일별 자본 차이 분포 ───')
    print(f'    평균 절대값 : ${daily_diffs.abs().mean():.0f}')
    print(f'    최대 절대값 : ${daily_diffs.abs().max():.0f}')


# ============================================================
# Spot-Perp 베이시스 분석
# ============================================================
print()
print('=' * 72)
print('Spot-Perp 베이시스 (1h 단위)')
print('=' * 72)
for coin, df_1h in [('BTC', btc_1h), ('ETH', eth_1h)]:
    basis = (df_1h['perp'] - df_1h['spot']) / df_1h['spot']
    print(f'\n{coin}:')
    print(f'  평균: {basis.mean()*100:+.4f}%')
    print(f'  std : {basis.std()*100:.4f}%')
    print(f'  [최소, 최대]: [{basis.min()*100:+.3f}%, {basis.max()*100:+.3f}%]')
    print(f'  베이시스 변동성이 추적오차 원인 — 일별 모델에선 일단위로만 측정됨')


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

color_p = 'darkred'
color_d = 'steelblue'

# (1, full) Equity 비교 - BTC
ax = fig.add_subplot(gs[0, :])
btc_p = results['BTC']['precise_obj']
btc_d = results['BTC']['daily_obj']
ax.plot(btc_p.index, btc_p['equity'], color=color_p, lw=1.5,
        label=f'BTC precise 1h → ${results["BTC"]["precise_final"]:,.0f}', alpha=0.8)
ax.plot(btc_d.index, btc_d['equity'], color=color_d, lw=2, ls='--',
        label=f'BTC daily model → ${results["BTC"]["daily_final"]:,.0f}', alpha=0.9)
ax.axhline(INITIAL, color='black', ls=':', alpha=0.5)
ax.set_ylabel('Equity (USD)')
ax.set_title(f'BTC: precise 1h sim vs daily model '
             f'(gap ${results["BTC"]["gap_abs"]:+,.0f})')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.grid(alpha=0.3)

# (2, 1) ETH equity
ax = fig.add_subplot(gs[1, 0])
eth_p = results['ETH']['precise_obj']
eth_d = results['ETH']['daily_obj']
ax.plot(eth_p.index, eth_p['equity'], color=color_p, lw=1.5,
        label=f'precise 1h', alpha=0.8)
ax.plot(eth_d.index, eth_d['equity'], color=color_d, lw=2, ls='--',
        label=f'daily', alpha=0.9)
ax.axhline(INITIAL, color='black', ls=':', alpha=0.5)
ax.set_ylabel('Equity')
ax.set_title(f'ETH equity (gap ${results["ETH"]["gap_abs"]:+,.0f})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 2) 일별 자본 차이
ax = fig.add_subplot(gs[1, 1])
for coin, color in [('BTC', 'darkorange'), ('ETH', 'steelblue')]:
    p = results[coin]['precise_obj']
    d = results[coin]['daily_obj']
    p_eod = p['equity'].resample('D').last().dropna()
    common = p_eod.index.intersection(d['equity'].index)
    diff = p_eod.loc[common] - d['equity'].loc[common]
    ax.plot(diff.index, diff, color=color, lw=1.2, label=f'{coin} (precise - daily)')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Equity gap ($)')
ax.set_xlabel('Date')
ax.set_title('Daily equity gap: precise - daily model')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (3, 1) Spot-Perp basis 시간 추이 BTC
ax = fig.add_subplot(gs[2, 0])
for coin, color in [('BTC', 'darkorange'), ('ETH', 'steelblue')]:
    df = btc_1h if coin == 'BTC' else eth_1h
    basis = (df['perp'] - df['spot']) / df['spot'] * 100
    ax.plot(basis.index, basis, color=color, alpha=0.5, lw=0.5, label=f'{coin} 1h')
    ma = basis.rolling(24).mean()
    ax.plot(ma.index, ma, color=color, lw=2, label=f'{coin} 24h MA')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Basis (%)')
ax.set_title('Spot-Perp basis (Perp - Spot) / Spot')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (3, 2) Funding PnL 누적 비교
ax = fig.add_subplot(gs[2, 1])
for coin, color in [('BTC', 'darkorange'), ('ETH', 'steelblue')]:
    p = results[coin]['precise_obj']
    d = results[coin]['daily_obj']
    ax.plot(p.index, p['cum_funding'], color=color, lw=2,
            label=f'{coin} precise funding')
    cum_d = d['funding_pnl'].cumsum()
    ax.plot(cum_d.index, cum_d, color=color, lw=1.5, ls='--', alpha=0.7,
            label=f'{coin} daily funding')
ax.set_ylabel('Cumulative funding PnL ($)')
ax.set_xlabel('Date')
ax.set_title('Funding income: precise (1h) vs daily aggregation')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

plt.suptitle('B-2 STEP 5: 1h precise simulation vs daily model validation',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'b2_precise_vs_daily.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 결론
# ============================================================
print()
print('=' * 72)
print('결론 — 일별 모델 정확성 평가')
print('=' * 72)
print()

for coin in ['BTC', 'ETH']:
    r = results[coin]
    rel_gap = abs(r['gap_pct']) / abs(r['daily_return']*100) if r['daily_return'] != 0 else 0
    print(f'  {coin}:')
    print(f'    절대 차이 (7개월): ${r["gap_abs"]:+.0f}')
    print(f'    상대 차이        : {r["gap_pct"]:+.3f}%p of initial')
    print(f'    일별 모델 수익률 : {r["daily_return"]*100:+.2f}%')
    print(f'    상대 오차        : {rel_gap*100:.1f}% of return')

    if abs(r['gap_pct']) < 0.05:
        print(f'    → ✓✓ 일별 모델 매우 정확 (오차 < 0.05%p)')
    elif abs(r['gap_pct']) < 0.2:
        print(f'    → ✓ 일별 모델 충분히 정확 (오차 < 0.2%p)')
    else:
        print(f'    → ⚠ 일별 모델 편향 큼 — 정밀 시뮬 필요')

# 저장
summary = {coin: {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
           for coin, r in results.items()}
with open('/tmp/mvp_b2_precise.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print()
print('데이터 저장: /tmp/mvp_b2_precise.json')
