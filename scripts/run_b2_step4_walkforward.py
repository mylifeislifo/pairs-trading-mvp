"""
B-2 STEP 4: Walk-forward 검증

설계:
  Train 180d + Test 90d + Step 30d → 약 28 윈도우
  
변형:
  (A) Buy-and-hold: 항상 진입 (현재 baseline)
  (B) Regime-aware: Train의 funding 평균이 양수일 때만 Test 진입

평가:
  - 각 윈도우 Test 기간 OOS Sharpe/Return/MDD
  - 분포 통계 (평균, 중앙값, std, 양수 비율)
  - 부트스트랩 95% CI
  - Train→Test 신호 가치 (regime-aware의 효과)

9-D 함정 회피 점검:
  - 단일 백테스트 +6.84% / Sharpe +12.71이 견고한가?
  - 부트스트랩 CI가 0을 포함하지 않는가?
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
# 데이터 준비 (step 3와 동일)
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
print(f'BTC: {len(btc)}일, ETH: {len(eth)}일')


# ============================================================
# 윈도우 정의
# ============================================================
TRAIN_DAYS = 180
TEST_DAYS = 90
STEP_DAYS = 30
INITIAL = 100_000
SPOT_FEE = 0.0008
PERP_FEE = 0.0004


def make_windows(df):
    ws = []
    i = 0
    while i + TRAIN_DAYS + TEST_DAYS <= len(df):
        ws.append({
            'idx': len(ws),
            'train_loc': (i, i + TRAIN_DAYS),
            'test_loc': (i + TRAIN_DAYS, i + TRAIN_DAYS + TEST_DAYS),
            'train_range': (df.index[i], df.index[i + TRAIN_DAYS - 1]),
            'test_range': (df.index[i + TRAIN_DAYS], df.index[i + TRAIN_DAYS + TEST_DAYS - 1]),
        })
        i += STEP_DAYS
    return ws


def test_backtest(df, train_funding_mean, regime_aware: bool):
    """
    Test 기간만 백테스트. 진입 가정: Test 시작일.
    regime_aware=True면 train_funding_mean > 0일 때만 진입.
    """
    if regime_aware and train_funding_mean <= 0:
        # 현금 보유 — 결과는 0
        return {
            'final_equity': INITIAL,
            'total_ret': 0.0,
            'sharpe': 0.0,
            'mdd': 0.0,
            'funding_pnl': 0.0,
            'tracking_error': 0.0,
            'entered': False,
        }

    notional = INITIAL / 2
    spot_pnl = df['spot_ret'] * notional
    perp_pnl = -df['perp_ret'] * notional
    funding_pnl = df['funding_daily'] * notional
    daily = spot_pnl + perp_pnl + funding_pnl

    # 거래비용: 진입(첫날) + 청산(마지막날)
    cost_each = (SPOT_FEE + PERP_FEE) * notional * 2  # 양다리
    daily.iloc[0] -= cost_each
    daily.iloc[-1] -= cost_each

    eq = INITIAL + daily.cumsum()
    days = len(eq)
    total_ret = eq.iloc[-1] / INITIAL - 1
    # 일일 수익률 → 연환산 Sharpe (90일짜리지만 비교 위해 연환산 처리)
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365) if daily_ret.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())

    return {
        'final_equity': float(eq.iloc[-1]),
        'total_ret': float(total_ret),
        'sharpe': float(sharpe),
        'mdd': mdd,
        'funding_pnl': float(funding_pnl.sum()),
        'tracking_error': float((spot_pnl + perp_pnl).sum()),
        'entered': True,
    }


# ============================================================
# Walk-forward 실행 (BTC, ETH)
# ============================================================
print()
print('=' * 72)
print('Walk-forward 실행')
print('=' * 72)

all_results = {}

for coin, df in [('BTC', btc), ('ETH', eth)]:
    windows = make_windows(df)
    print(f'\n{coin}: {len(windows)} 윈도우')

    coin_results = []
    for w in windows:
        train = df.iloc[w['train_loc'][0]:w['train_loc'][1]]
        test = df.iloc[w['test_loc'][0]:w['test_loc'][1]]
        train_fund_mean = train['funding_daily'].mean()

        # (A) Buy-and-hold
        res_bh = test_backtest(test, train_fund_mean, regime_aware=False)
        # (B) Regime-aware
        res_ra = test_backtest(test, train_fund_mean, regime_aware=True)

        coin_results.append({
            'window': w['idx'],
            'train_range': w['train_range'],
            'test_range': w['test_range'],
            'train_fund_mean': float(train_fund_mean),
            'test_fund_mean': float(test['funding_daily'].mean()),
            'bh': res_bh,
            'ra': res_ra,
        })

    all_results[coin] = coin_results

    # 출력
    print(f'  {"W":>3s}  {"Test 시작":<11s} {"Train fd%":>10s} {"Test fd%":>10s} '
          f'{"BH Ret":>8s} {"BH Sh":>7s} {"RA Ret":>8s} {"RA Sh":>7s} {"RA진입":>8s}')
    for r in coin_results:
        print(f'  W{r["window"]:>2d}  {r["test_range"][0].date()!s:<11s} '
              f'{r["train_fund_mean"]*100:>+9.4f}% '
              f'{r["test_fund_mean"]*100:>+9.4f}% '
              f'{r["bh"]["total_ret"]*100:>+7.2f}% '
              f'{r["bh"]["sharpe"]:>+7.2f} '
              f'{r["ra"]["total_ret"]*100:>+7.2f}% '
              f'{r["ra"]["sharpe"]:>+7.2f} '
              f'{"진입" if r["ra"]["entered"] else "현금":>8s}')


# ============================================================
# 통계 분석
# ============================================================
print()
print('=' * 72)
print('분포 통계')
print('=' * 72)


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


summary = {}
for coin in ['BTC', 'ETH']:
    rs = all_results[coin]
    bh_rets = np.array([r['bh']['total_ret'] for r in rs])
    bh_shs = np.array([r['bh']['sharpe'] for r in rs])
    ra_rets = np.array([r['ra']['total_ret'] for r in rs])
    ra_shs = np.array([r['ra']['sharpe'] for r in rs])
    train_fds = np.array([r['train_fund_mean'] for r in rs])
    test_fds = np.array([r['test_fund_mean'] for r in rs])

    bh_pos = (bh_shs > 0).mean() * 100
    ra_pos = (ra_shs > 0).mean() * 100
    n_entered = sum(1 for r in rs if r['ra']['entered'])

    bh_sh_ci = bootstrap_ci(bh_shs)
    bh_ret_ci = bootstrap_ci(bh_rets)
    ra_sh_ci = bootstrap_ci(ra_shs)
    ra_ret_ci = bootstrap_ci(ra_rets)

    # Train→Test 펀딩비 상관
    corr = np.corrcoef(train_fds, test_fds)[0, 1]

    print(f'\n{coin} (n={len(rs)}):')
    print(f'  (A) Buy-and-hold:')
    print(f'    평균 Return  : {bh_rets.mean()*100:+.2f}%  중앙값 {np.median(bh_rets)*100:+.2f}%  std {bh_rets.std()*100:.2f}%')
    print(f'    평균 Sharpe  : {bh_shs.mean():+.2f}  중앙값 {np.median(bh_shs):+.2f}  std {bh_shs.std():.2f}')
    print(f'    양수 비율     : {bh_pos:.1f}%')
    print(f'    95% CI Sharpe: [{bh_sh_ci[0]:+.2f}, {bh_sh_ci[1]:+.2f}]')
    print(f'    95% CI Return: [{bh_ret_ci[0]*100:+.2f}%, {bh_ret_ci[1]*100:+.2f}%]')

    print(f'  (B) Regime-aware ({n_entered}/{len(rs)} 진입):')
    print(f'    평균 Return  : {ra_rets.mean()*100:+.2f}%  중앙값 {np.median(ra_rets)*100:+.2f}%')
    print(f'    평균 Sharpe  : {ra_shs.mean():+.2f}  std {ra_shs.std():.2f}')
    print(f'    양수 비율     : {ra_pos:.1f}%')
    print(f'    95% CI Sharpe: [{ra_sh_ci[0]:+.2f}, {ra_sh_ci[1]:+.2f}]')
    print(f'    95% CI Return: [{ra_ret_ci[0]*100:+.2f}%, {ra_ret_ci[1]*100:+.2f}%]')

    print(f'  Train→Test 펀딩비 상관계수: {corr:+.3f}')

    summary[coin] = {
        'n_windows': len(rs),
        'bh': {
            'mean_return': float(bh_rets.mean()),
            'median_return': float(np.median(bh_rets)),
            'mean_sharpe': float(bh_shs.mean()),
            'median_sharpe': float(np.median(bh_shs)),
            'std_sharpe': float(bh_shs.std()),
            'positive_pct': float(bh_pos),
            'ci_sharpe': list(bh_sh_ci),
            'ci_return': list(bh_ret_ci),
        },
        'ra': {
            'mean_return': float(ra_rets.mean()),
            'median_return': float(np.median(ra_rets)),
            'mean_sharpe': float(ra_shs.mean()),
            'positive_pct': float(ra_pos),
            'ci_sharpe': list(ra_sh_ci),
            'ci_return': list(ra_ret_ci),
            'n_entered': int(n_entered),
        },
        'train_test_correlation': float(corr),
    }


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.30)

color_btc = 'darkorange'
color_eth = 'steelblue'

# (1, full) 윈도우별 Sharpe (BTC + ETH 같이)
ax = fig.add_subplot(gs[0, :])
for coin, color in [('BTC', color_btc), ('ETH', color_eth)]:
    rs = all_results[coin]
    x = [r['window'] for r in rs]
    shs = [r['bh']['sharpe'] for r in rs]
    ax.plot(x, shs, 'o-', color=color, lw=1.5, markersize=7, label=f'{coin} BH')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Window (test start month)')
ax.set_ylabel('Test Sharpe (annualized)')
ax.set_title('B-2 Walk-forward: out-of-sample Sharpe per window')
ax.legend()
ax.grid(alpha=0.3)
# X축에 날짜 표시 (대략)
btc_results = all_results['BTC']
xtick_pos = list(range(0, len(btc_results), 4))
xtick_labels = [btc_results[i]['test_range'][0].strftime('%Y-%m') for i in xtick_pos]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labels)

# (2, 1) Sharpe boxplot BTC
ax = fig.add_subplot(gs[1, 0])
data_box = [
    [r['bh']['sharpe'] for r in all_results['BTC']],
    [r['ra']['sharpe'] for r in all_results['BTC']],
    [r['bh']['sharpe'] for r in all_results['ETH']],
    [r['ra']['sharpe'] for r in all_results['ETH']],
]
labels_box = ['BTC BH', 'BTC RA', 'ETH BH', 'ETH RA']
bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)
colors_box = [color_btc, color_btc, color_eth, color_eth]
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.axhline(0, color='red', ls='--', alpha=0.7)
for i, d in enumerate(data_box):
    ax.scatter([i + 1], [np.mean(d)], color='red', marker='D', s=80, zorder=3)
ax.set_ylabel('Test Sharpe distribution')
ax.set_title('Sharpe distribution across windows')
ax.grid(alpha=0.3, axis='y')

# (2, 2) Mean Sharpe with 95% CI
ax = fig.add_subplot(gs[1, 1])
variants = ['BTC BH', 'BTC RA', 'ETH BH', 'ETH RA']
means = [summary['BTC']['bh']['mean_sharpe'], summary['BTC']['ra']['mean_sharpe'],
         summary['ETH']['bh']['mean_sharpe'], summary['ETH']['ra']['mean_sharpe']]
cis = [summary['BTC']['bh']['ci_sharpe'], summary['BTC']['ra']['ci_sharpe'],
       summary['ETH']['bh']['ci_sharpe'], summary['ETH']['ra']['ci_sharpe']]
errs_lo = [m - ci[0] for m, ci in zip(means, cis)]
errs_hi = [ci[1] - m for m, ci in zip(means, cis)]
ax.errorbar(range(4), means, yerr=[errs_lo, errs_hi],
            fmt='o', capsize=10, capthick=2, lw=2, markersize=12,
            color='black', ecolor='gray')
for i, c in enumerate(colors_box):
    ax.scatter([i], [means[i]], color=c, s=200, zorder=3,
               edgecolor='black', lw=1.5)
ax.axhline(0, color='red', lw=1, ls='--', alpha=0.7, label='Zero alpha')
ax.set_xticks(range(4))
ax.set_xticklabels(variants)
ax.set_ylabel('Mean Sharpe + 95% CI')
ax.set_title('Mean Sharpe with bootstrap 95% CI')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (3, 1) 양수 비율
ax = fig.add_subplot(gs[2, 0])
pos_vals = [summary['BTC']['bh']['positive_pct'], summary['BTC']['ra']['positive_pct'],
            summary['ETH']['bh']['positive_pct'], summary['ETH']['ra']['positive_pct']]
bars = ax.bar(variants, pos_vals, color=colors_box, alpha=0.85)
ax.axhline(50, color='red', ls='--', alpha=0.7, label='50% (random)')
ax.axhline(60, color='orange', ls=':', alpha=0.7, label='60% threshold')
for bar, v in zip(bars, pos_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f'{v:.1f}%',
            ha='center', fontweight='bold')
ax.set_ylim(0, 105)
ax.set_ylabel('% windows with positive Sharpe')
ax.set_title('Positive-Sharpe window ratio')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 2) Train→Test 펀딩비 상관 (regime 신호 가치)
ax = fig.add_subplot(gs[2, 1])
for coin, color in [('BTC', color_btc), ('ETH', color_eth)]:
    rs = all_results[coin]
    train_f = [r['train_fund_mean']*100 for r in rs]
    test_f = [r['test_fund_mean']*100 for r in rs]
    ax.scatter(train_f, test_f, color=color, alpha=0.7, s=80, label=coin)
    # 회귀선
    z = np.polyfit(train_f, test_f, 1)
    x_line = np.linspace(min(train_f), max(train_f), 50)
    ax.plot(x_line, np.poly1d(z)(x_line), color=color, ls='--', alpha=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('Train period mean daily funding (%)')
ax.set_ylabel('Test period mean daily funding (%)')
ax.set_title(f'Regime signal value: Train fd → Test fd '
             f'(corr BTC={summary["BTC"]["train_test_correlation"]:+.2f}, '
             f'ETH={summary["ETH"]["train_test_correlation"]:+.2f})')
ax.legend()
ax.grid(alpha=0.3)

# (4, full) Equity curve 누적 — 윈도우 결과 이어붙이기
ax = fig.add_subplot(gs[3, :])
for coin, color in [('BTC', color_btc), ('ETH', color_eth)]:
    rs = all_results[coin]
    # 각 윈도우 test 시작일 + Sharpe로 시각화
    dates = [r['test_range'][0] for r in rs]
    rets = [r['bh']['total_ret']*100 for r in rs]
    ax.bar(dates, rets, width=20, color=color, alpha=0.7, label=f'{coin} BH test return')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Test window start')
ax.set_ylabel('Test return (%)')
ax.set_title('Test-period returns over time (3-month OOS)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3, axis='y')

plt.suptitle('B-2 Walk-forward: delta-neutral funding arbitrage validation',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'b2_walkforward.png'))
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


def verdict(stats):
    lo, hi = stats['ci_sharpe']
    pos = stats['positive_pct']
    if lo > 0:
        return f'✓✓✓ 진짜 alpha (Sharpe CI [{lo:+.2f}, {hi:+.2f}] 양수)'
    elif lo <= 0 <= hi:
        return f'⚠ 결론 불가 (CI에 0 포함)'
    else:
        return f'✗ 음수 alpha'


for coin in ['BTC', 'ETH']:
    s = summary[coin]
    print(f'  {coin}:')
    print(f'    BH 평결: {verdict(s["bh"])}')
    print(f'    RA 평결: {verdict(s["ra"])}')
    print(f'    Regime 신호 가치: Train→Test 펀딩 상관 {s["train_test_correlation"]:+.2f}')

# 9차와 비교
print()
print('  비교 (9-C 페어 트레이딩):')
print(f'    9-C (b) Sharpe CI [-0.61, +0.16] ⚠ 0 포함')
print(f'    9-C r180,cd=30 Sharpe CI [-0.44, -0.03] ✗ 음수')
print()
print(f'    B-2 BTC BH Sharpe CI [{summary["BTC"]["bh"]["ci_sharpe"][0]:+.2f}, '
      f'{summary["BTC"]["bh"]["ci_sharpe"][1]:+.2f}]')

# 저장
with open('/tmp/mvp_b2_walkforward.json', 'w') as f:
    json.dump(summary, f, indent=2)
print()
print('데이터 저장: /tmp/mvp_b2_walkforward.json')
