"""
B-2 STEP 6: 편향 보정 후 재해석

Step 5에서 발견:
  - 일별 모델은 funding을 평균 22.5% 과대평가
    (BTC -14.4%, ETH -30.7%)
  - 보수적으로 25% 일괄 차감

이 보정을 step 3 (전체 기간) + step 4 (walk-forward) 결과에 적용.
"""
import os, pickle, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))

FUNDING_BIAS_CORRECTION = 0.25  # 보수적 25% 차감
INITIAL = 100_000
SPOT_FEE = 0.0008
PERP_FEE = 0.0004


# ============================================================
# 데이터 준비
# ============================================================
def prep_data(coin: str):
    spot = cache[f'{coin}_spot_1d'][['close']].rename(columns={'close': 'spot'})
    perp = cache[f'{coin}_perp_1d'][['close']].rename(columns={'close': 'perp'})
    fund_hourly = cache[f'HL_{coin}_funding'][['fundingRate']]
    spot.index = spot.index.normalize()
    perp.index = perp.index.normalize()
    fund_hourly = fund_hourly.copy()
    fund_hourly['date'] = fund_hourly.index.normalize()
    fund_daily = fund_hourly.groupby('date')['fundingRate'].sum()
    fund_daily.name = 'funding_daily'
    fund_daily.index.name = None
    df = spot.join(perp, how='inner').join(fund_daily, how='inner').dropna()
    df['spot_ret'] = df['spot'].pct_change()
    df['perp_ret'] = df['perp'].pct_change()
    return df.dropna()


btc = prep_data('BTC')
eth = prep_data('ETH')


# ============================================================
# 보정된 백테스트
# ============================================================
def backtest_corrected(df: pd.DataFrame, correction: float = FUNDING_BIAS_CORRECTION):
    notional = INITIAL / 2
    spot_pnl = df['spot_ret'] * notional
    perp_pnl = -df['perp_ret'] * notional
    # ★ 보정 적용
    funding_pnl_raw = df['funding_daily'] * notional
    funding_pnl_corrected = funding_pnl_raw * (1 - correction)

    daily_raw = spot_pnl + perp_pnl + funding_pnl_raw
    daily_corr = spot_pnl + perp_pnl + funding_pnl_corrected

    cost = (SPOT_FEE + PERP_FEE) * notional * 2
    daily_raw.iloc[0] -= cost
    daily_raw.iloc[-1] -= cost
    daily_corr.iloc[0] -= cost
    daily_corr.iloc[-1] -= cost

    eq_raw = INITIAL + daily_raw.cumsum()
    eq_corr = INITIAL + daily_corr.cumsum()

    return pd.DataFrame({
        'equity_raw': eq_raw,
        'equity_corrected': eq_corr,
        'spot_pnl': spot_pnl,
        'perp_pnl': perp_pnl,
        'funding_pnl_raw': funding_pnl_raw,
        'funding_pnl_corrected': funding_pnl_corrected,
    })


def metrics(eq):
    days = len(eq)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    annual = (1 + total_ret) ** (365 / days) - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return {
        'total_return': float(total_ret), 'annual': float(annual),
        'sharpe': float(sharpe), 'mdd': mdd, 'days': days,
        'final': float(eq.iloc[-1]),
    }


# ============================================================
# 전체 기간 보정 결과
# ============================================================
print('=' * 72)
print('전체 기간 보정 결과')
print('=' * 72)

full_results = {}
for coin, df in [('BTC', btc), ('ETH', eth)]:
    res = backtest_corrected(df)
    m_raw = metrics(res['equity_raw'])
    m_corr = metrics(res['equity_corrected'])
    full_results[coin] = {'raw': m_raw, 'corr': m_corr, 'res': res}

    print(f'\n{coin}:')
    print(f'  {"지표":<15s} {"원본 일별":>15s} {"보정 25%":>15s} {"변화":>12s}')
    print(f'  {"-"*15} {"-"*15} {"-"*15} {"-"*12}')
    print(f'  {"최종 자본":<15s} ${m_raw["final"]:>13,.0f}  ${m_corr["final"]:>13,.0f}  ${m_corr["final"]-m_raw["final"]:>+10,.0f}')
    print(f'  {"수익률":<15s} {m_raw["total_return"]*100:>+13.2f}%  {m_corr["total_return"]*100:>+13.2f}%  {(m_corr["total_return"]-m_raw["total_return"])*100:>+10.2f}%p')
    print(f'  {"연환산":<15s} {m_raw["annual"]*100:>+13.2f}%  {m_corr["annual"]*100:>+13.2f}%  {(m_corr["annual"]-m_raw["annual"])*100:>+10.2f}%p')
    print(f'  {"Sharpe":<15s} {m_raw["sharpe"]:>+13.2f}   {m_corr["sharpe"]:>+13.2f}   {m_corr["sharpe"]-m_raw["sharpe"]:>+10.2f}')
    print(f'  {"MDD":<15s} {m_raw["mdd"]*100:>+13.2f}%  {m_corr["mdd"]*100:>+13.2f}%  {(m_corr["mdd"]-m_raw["mdd"])*100:>+10.2f}%p')


# ============================================================
# 연도별 보정
# ============================================================
print()
print('=' * 72)
print('연도별 보정 결과')
print('=' * 72)
print()
print(f'  {"":<5s} {"":<5s} {"원본":>20s} {"보정 25%":>20s}')
print(f'  {"코인":<5s} {"연":<5s} {"수익률":>10s} {"Sharpe":>8s}  {"수익률":>10s} {"Sharpe":>8s}')
print(f'  {"-"*5} {"-"*5} {"-"*10} {"-"*8}  {"-"*10} {"-"*8}')

annual_results = {}
for coin in ['BTC', 'ETH']:
    annual_results[coin] = []
    res = full_results[coin]['res']
    for year in [2023, 2024, 2025, 2026]:
        mask = res.index.year == year
        if mask.sum() < 10:
            continue
        sub = res[mask]
        eq_raw = sub['equity_raw']
        eq_corr = sub['equity_corrected']
        # 연도 내 시작 자본 → 끝
        days = len(sub)
        r_raw = eq_raw.iloc[-1] / eq_raw.iloc[0] - 1
        r_corr = eq_corr.iloc[-1] / eq_corr.iloc[0] - 1
        ann_raw = (1 + r_raw) ** (365 / days) - 1
        ann_corr = (1 + r_corr) ** (365 / days) - 1
        d_raw = eq_raw.pct_change().dropna()
        d_corr = eq_corr.pct_change().dropna()
        sh_raw = d_raw.mean() / d_raw.std() * np.sqrt(365) if d_raw.std() > 0 else 0
        sh_corr = d_corr.mean() / d_corr.std() * np.sqrt(365) if d_corr.std() > 0 else 0
        annual_results[coin].append({
            'year': year, 'raw_ret': r_raw, 'raw_annual': ann_raw, 'raw_sharpe': sh_raw,
            'corr_ret': r_corr, 'corr_annual': ann_corr, 'corr_sharpe': sh_corr,
        })
        print(f'  {coin:<5s} {year:<5d} {ann_raw*100:>+9.2f}% {sh_raw:>+8.2f}   '
              f'{ann_corr*100:>+9.2f}% {sh_corr:>+8.2f}')


# ============================================================
# Walk-forward 보정
# ============================================================
print()
print('=' * 72)
print('Walk-forward 보정')
print('=' * 72)

TRAIN_DAYS = 180
TEST_DAYS = 90
STEP_DAYS = 30


def make_windows(df):
    ws = []
    i = 0
    while i + TRAIN_DAYS + TEST_DAYS <= len(df):
        ws.append({
            'idx': len(ws),
            'train_loc': (i, i + TRAIN_DAYS),
            'test_loc': (i + TRAIN_DAYS, i + TRAIN_DAYS + TEST_DAYS),
            'test_range': (df.index[i + TRAIN_DAYS],
                          df.index[i + TRAIN_DAYS + TEST_DAYS - 1]),
        })
        i += STEP_DAYS
    return ws


def test_window_corrected(df, correction):
    notional = INITIAL / 2
    spot_pnl = df['spot_ret'] * notional
    perp_pnl = -df['perp_ret'] * notional
    funding_pnl = df['funding_daily'] * notional * (1 - correction)
    daily = spot_pnl + perp_pnl + funding_pnl
    cost = (SPOT_FEE + PERP_FEE) * notional * 2
    daily.iloc[0] -= cost
    daily.iloc[-1] -= cost
    eq = INITIAL + daily.cumsum()
    total_ret = eq.iloc[-1] / INITIAL - 1
    d_ret = eq.pct_change().dropna()
    sharpe = d_ret.mean() / d_ret.std() * np.sqrt(365) if d_ret.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return {'total_ret': float(total_ret), 'sharpe': float(sharpe), 'mdd': mdd}


def bootstrap_ci(arr, n_boot=10000):
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


wf_results = {}
for coin, df in [('BTC', btc), ('ETH', eth)]:
    windows = make_windows(df)
    raw_shs, corr_shs = [], []
    raw_rets, corr_rets = [], []
    for w in windows:
        test = df.iloc[w['test_loc'][0]:w['test_loc'][1]]
        r_raw = test_window_corrected(test, 0)
        r_corr = test_window_corrected(test, FUNDING_BIAS_CORRECTION)
        raw_shs.append(r_raw['sharpe'])
        corr_shs.append(r_corr['sharpe'])
        raw_rets.append(r_raw['total_ret'])
        corr_rets.append(r_corr['total_ret'])

    raw_shs, corr_shs = np.array(raw_shs), np.array(corr_shs)
    raw_rets, corr_rets = np.array(raw_rets), np.array(corr_rets)

    raw_ci = bootstrap_ci(raw_shs)
    corr_ci = bootstrap_ci(corr_shs)

    wf_results[coin] = {
        'windows': windows,
        'raw_shs': raw_shs, 'corr_shs': corr_shs,
        'raw_rets': raw_rets, 'corr_rets': corr_rets,
        'raw_ci': raw_ci, 'corr_ci': corr_ci,
    }

    print(f'\n{coin} ({len(windows)} 윈도우):')
    print(f'  {"":<15s} {"원본":>20s} {"보정 25%":>20s}')
    print(f'  {"평균 Sharpe":<15s} {raw_shs.mean():>+19.2f} {corr_shs.mean():>+19.2f}')
    print(f'  {"중앙 Sharpe":<15s} {np.median(raw_shs):>+19.2f} {np.median(corr_shs):>+19.2f}')
    print(f'  {"양수 비율":<15s} {(raw_shs > 0).mean()*100:>18.1f}% {(corr_shs > 0).mean()*100:>18.1f}%')
    print(f'  {"95% CI":<15s} [{raw_ci[0]:+.2f}, {raw_ci[1]:+.2f}]  '
          f'[{corr_ci[0]:+.2f}, {corr_ci[1]:+.2f}]')
    print(f'  {"평균 Test ret":<15s} {raw_rets.mean()*100:>+18.2f}% {corr_rets.mean()*100:>+18.2f}%')


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full) Raw vs Corrected equity curves
ax = fig.add_subplot(gs[0, :])
for coin, color in [('BTC', 'darkorange'), ('ETH', 'steelblue')]:
    r = full_results[coin]
    ax.plot(r['res'].index, r['res']['equity_raw'],
            color=color, lw=1.5, ls='--', alpha=0.6,
            label=f'{coin} raw  → {r["raw"]["annual"]*100:+.2f}%/yr  Sh {r["raw"]["sharpe"]:+.2f}')
    ax.plot(r['res'].index, r['res']['equity_corrected'],
            color=color, lw=2,
            label=f'{coin} corrected → {r["corr"]["annual"]*100:+.2f}%/yr  Sh {r["corr"]["sharpe"]:+.2f}')
ax.axhline(INITIAL, color='black', ls=':', alpha=0.5)
ax.set_ylabel('Equity (USD)')
ax.set_title('Full-period equity: raw daily model vs bias-corrected (-25% funding)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2, 1) Walk-forward Sharpe per window
ax = fig.add_subplot(gs[1, 0])
for coin, color in [('BTC', 'darkorange'), ('ETH', 'steelblue')]:
    r = wf_results[coin]
    x = list(range(len(r['windows'])))
    ax.plot(x, r['raw_shs'], 'o--', color=color, alpha=0.5, lw=1, markersize=5,
            label=f'{coin} raw')
    ax.plot(x, r['corr_shs'], 's-', color=color, lw=1.8, markersize=7,
            label=f'{coin} corrected')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Walk-forward window')
ax.set_ylabel('Test Sharpe')
ax.set_title('Walk-forward Sharpe: raw vs corrected')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (2, 2) Mean Sharpe ± CI 비교
ax = fig.add_subplot(gs[1, 1])
labels = ['BTC raw', 'BTC corr', 'ETH raw', 'ETH corr']
means = [wf_results['BTC']['raw_shs'].mean(),
         wf_results['BTC']['corr_shs'].mean(),
         wf_results['ETH']['raw_shs'].mean(),
         wf_results['ETH']['corr_shs'].mean()]
cis = [wf_results['BTC']['raw_ci'], wf_results['BTC']['corr_ci'],
       wf_results['ETH']['raw_ci'], wf_results['ETH']['corr_ci']]
errs_lo = [m - ci[0] for m, ci in zip(means, cis)]
errs_hi = [ci[1] - m for m, ci in zip(means, cis)]
ax.errorbar(range(4), means, yerr=[errs_lo, errs_hi],
            fmt='o', capsize=10, capthick=2, lw=2, markersize=12,
            color='black', ecolor='gray')
colors_seq = ['darkorange', 'darkorange', 'steelblue', 'steelblue']
markers_seq = ['o', 's', 'o', 's']
for i, (c, m_sym) in enumerate(zip(colors_seq, markers_seq)):
    ax.scatter([i], [means[i]], color=c, s=200, zorder=3,
               edgecolor='black', lw=1.5, marker=m_sym)
ax.axhline(0, color='red', lw=1, ls='--', alpha=0.7, label='Zero alpha')
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_ylabel('Mean Sharpe ± 95% CI')
ax.set_title('Walk-forward CI: raw vs corrected')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (3, 1) 연도별 비교
ax = fig.add_subplot(gs[2, 0])
years = sorted(set(r['year'] for r in annual_results['BTC']))
x = np.arange(len(years))
w_bar = 0.2
btc_raw = [next((r['raw_annual']*100 for r in annual_results['BTC'] if r['year']==y), 0) for y in years]
btc_corr = [next((r['corr_annual']*100 for r in annual_results['BTC'] if r['year']==y), 0) for y in years]
eth_raw = [next((r['raw_annual']*100 for r in annual_results['ETH'] if r['year']==y), 0) for y in years]
eth_corr = [next((r['corr_annual']*100 for r in annual_results['ETH'] if r['year']==y), 0) for y in years]
ax.bar(x - 1.5*w_bar, btc_raw, w_bar, color='darkorange', alpha=0.5, label='BTC raw')
ax.bar(x - 0.5*w_bar, btc_corr, w_bar, color='darkorange', label='BTC corr')
ax.bar(x + 0.5*w_bar, eth_raw, w_bar, color='steelblue', alpha=0.5, label='ETH raw')
ax.bar(x + 1.5*w_bar, eth_corr, w_bar, color='steelblue', label='ETH corr')
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylabel('Annualized return (%)')
ax.set_title('Annualized return by year: raw vs corrected')
ax.axhline(0, color='black', lw=0.5)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, axis='y')

# (3, 2) Funding contribution decomposition
ax = fig.add_subplot(gs[2, 1])
labels = ['BTC', 'ETH']
funding_raw = [full_results[c]['res']['funding_pnl_raw'].sum() for c in labels]
funding_corr = [full_results[c]['res']['funding_pnl_corrected'].sum() for c in labels]
costs = [-2 * (SPOT_FEE + PERP_FEE) * INITIAL  for _ in labels]
tracking = [(full_results[c]['res']['spot_pnl'].sum() +
             full_results[c]['res']['perp_pnl'].sum()) for c in labels]
x = np.arange(len(labels))
ax.bar(x - 0.2, funding_raw, 0.18, color='green', alpha=0.5, label='Funding raw')
ax.bar(x - 0.0, funding_corr, 0.18, color='green', label='Funding corr')
ax.bar(x + 0.2, tracking, 0.18, color='red', alpha=0.7, label='Tracking err')
ax.bar(x + 0.4, costs, 0.18, color='gray', label='Fees')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('PnL ($)')
ax.set_title('Full-period PnL decomposition')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('B-2 STEP 6: bias-corrected reanalysis (funding × 0.75)',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'b2_corrected.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 결론
# ============================================================
print()
print('=' * 72)
print('결론 — 보정 후 평결')
print('=' * 72)
print()


def verdict(ci):
    lo, hi = ci
    if lo > 0:
        return f'✓✓✓ 진짜 alpha (95% CI 모두 양수)'
    elif lo <= 0 <= hi:
        return f'⚠ 결론 불가'
    else:
        return f'✗ 음수 alpha'


for coin in ['BTC', 'ETH']:
    raw_ci = wf_results[coin]['raw_ci']
    corr_ci = wf_results[coin]['corr_ci']
    print(f'  {coin}:')
    print(f'    원본 walk-forward CI : [{raw_ci[0]:+.2f}, {raw_ci[1]:+.2f}]  {verdict(raw_ci)}')
    print(f'    보정 walk-forward CI : [{corr_ci[0]:+.2f}, {corr_ci[1]:+.2f}]  {verdict(corr_ci)}')

# 9-C, 10-A와 비교
print()
print('  9차 시리즈 벤치마크와 비교:')
print(f'    페어 트레이딩 9-C (b)    : CI [-0.61, +0.16]  ⚠')
print(f'    페어 트레이딩 10-A 확장  : CI [-0.65, +0.32]  ⚠')
print(f'    B-2 BTC corr             : CI [{wf_results["BTC"]["corr_ci"][0]:+.2f}, '
      f'{wf_results["BTC"]["corr_ci"][1]:+.2f}]  ★')
print(f'    B-2 ETH corr             : CI [{wf_results["ETH"]["corr_ci"][0]:+.2f}, '
      f'{wf_results["ETH"]["corr_ci"][1]:+.2f}]  ★')

# Save
summary = {
    'bias_correction': FUNDING_BIAS_CORRECTION,
    'full_period': {
        coin: {
            'raw': full_results[coin]['raw'],
            'corr': full_results[coin]['corr'],
        } for coin in ['BTC', 'ETH']
    },
    'walkforward': {
        coin: {
            'n_windows': len(wf_results[coin]['windows']),
            'raw_mean_sharpe': float(wf_results[coin]['raw_shs'].mean()),
            'corr_mean_sharpe': float(wf_results[coin]['corr_shs'].mean()),
            'raw_pos_pct': float((wf_results[coin]['raw_shs'] > 0).mean() * 100),
            'corr_pos_pct': float((wf_results[coin]['corr_shs'] > 0).mean() * 100),
            'raw_ci': list(wf_results[coin]['raw_ci']),
            'corr_ci': list(wf_results[coin]['corr_ci']),
            'raw_mean_return': float(wf_results[coin]['raw_rets'].mean()),
            'corr_mean_return': float(wf_results[coin]['corr_rets'].mean()),
        } for coin in ['BTC', 'ETH']
    },
    'annual': annual_results,
}
with open('/tmp/mvp_b2_corrected.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print()
print('데이터 저장: /tmp/mvp_b2_corrected.json')
