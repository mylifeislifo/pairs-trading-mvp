"""
10차 MVP-A: STEP 3 — 분석 + 9-C (28종목) vs 10-A (99종목) 비교
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

CACHE_10A = '/tmp/mvp10a_cache.pkl'
CACHE_9C = '/tmp/mvp9c_cache.pkl'

cache_10a = pickle.load(open(CACHE_10A, 'rb'))
cache_9c = pickle.load(open(CACHE_9C, 'rb'))

windows = cache_10a['windows']
print(f'윈도우 수: {len(windows)}')


# 각 윈도우 결과 수집
def collect(cache, prefix='b_w'):
    rets, shs, mdds, pool_sizes, pool_fulls = [], [], [], [], []
    for w in windows:
        key = f'{prefix}{w["idx"]}'
        r = cache.get(key, {})
        rets.append(r.get('return', 0))
        shs.append(r.get('sharpe', 0))
        mdds.append(r.get('mdd', 0))
        pool_sizes.append(r.get('pool_size', 0))
        pool_fulls.append(r.get('pool_size_full', r.get('pool_size', 0)))
    return (np.array(rets), np.array(shs), np.array(mdds),
            np.array(pool_sizes), np.array(pool_fulls))


rets_10a, shs_10a, mdds_10a, sizes_10a, fulls_10a = collect(cache_10a)
rets_9c, shs_9c, mdds_9c, sizes_9c, _ = collect(cache_9c)


# ============================================================
# 통계 요약
# ============================================================
print()
print('=' * 80)
print('Walk-Forward 결과 — 9-C (28종목) vs 10-A (99종목)')
print('=' * 80)
print()


def summary_stats(name, rets, shs, mdds):
    mean_r = np.mean(rets)
    median_r = np.median(rets)
    mean_s = np.mean(shs)
    median_s = np.median(shs)
    std_s = np.std(shs)
    pos_pct = (shs > 0).mean() * 100
    mean_m = np.mean(mdds)
    print(f'  {name:<25s} mean Ret {mean_r*100:+.2f}%  median {median_r*100:+.2f}%  '
          f'mean Sh {mean_s:+.2f}  median {median_s:+.2f}  std {std_s:.2f}  '
          f'pos% {pos_pct:.1f}  MDD {mean_m*100:+.2f}%')
    return {
        'mean_ret': float(mean_r), 'median_ret': float(median_r),
        'mean_sharpe': float(mean_s), 'median_sharpe': float(median_s),
        'std_sharpe': float(std_s),
        'positive_pct': float(pos_pct),
        'mean_mdd': float(mean_m),
        'all_shs': shs.tolist(),
    }


stats_9c = summary_stats('(b) 9-C  (28 tickers) ', rets_9c, shs_9c, mdds_9c)
stats_10a = summary_stats('(b) 10-A (99 tickers)', rets_10a, shs_10a, mdds_10a)


# 부트스트랩 95% CI
def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


print()
print('부트스트랩 95% 신뢰구간:')
lo_9c, hi_9c = bootstrap_ci(shs_9c)
lo_10a, hi_10a = bootstrap_ci(shs_10a)
stats_9c['ci'] = [lo_9c, hi_9c]
stats_10a['ci'] = [lo_10a, hi_10a]

ci_9c_excl_zero = lo_9c > 0 or hi_9c < 0
ci_10a_excl_zero = lo_10a > 0 or hi_10a < 0
print(f'  9-C  (28 tickers) : Sh CI [{lo_9c:+.2f}, {hi_9c:+.2f}]  {"✓ 0 미포함" if ci_9c_excl_zero else "⚠ 0 포함"}')
print(f'  10-A (99 tickers) : Sh CI [{lo_10a:+.2f}, {hi_10a:+.2f}]  {"✓ 0 미포함" if ci_10a_excl_zero else "⚠ 0 포함"}')


# ============================================================
# 윈도우별 상세
# ============================================================
print()
print('=' * 80)
print('윈도우별 Sharpe 비교')
print('=' * 80)
print()
print(f'  {"W":>3s}  {"test 시작":<11s} {"9-C Sh":>8s} {"10-A Sh":>9s} {"Δ":>7s} '
      f'{"10-A pool":>10s}')
print(f'  {"-"*3}  {"-"*11} {"-"*8} {"-"*9} {"-"*7} {"-"*10}')

for w in windows:
    i = w['idx']
    delta = shs_10a[i] - shs_9c[i]
    print(f'  W{i:>2d}  {w["test_range"][0].date()!s:<11s} '
          f'{shs_9c[i]:>+7.2f}  '
          f'{shs_10a[i]:>+8.2f}  '
          f'{delta:>+6.2f}  '
          f'{sizes_10a[i]:>3d}/{fulls_10a[i]:<5d}')

print()
# 평균 Sharpe 향상
delta_mean = stats_10a['mean_sharpe'] - stats_9c['mean_sharpe']
print(f'  평균 Sharpe 변화: {stats_9c["mean_sharpe"]:+.2f} → {stats_10a["mean_sharpe"]:+.2f}  '
      f'({delta_mean:+.2f})')
print(f'  양수 비율 변화 : {stats_9c["positive_pct"]:.1f}% → {stats_10a["positive_pct"]:.1f}%')


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full) 윈도우별 Sharpe 비교
ax = fig.add_subplot(gs[0, :])
x = np.arange(len(windows))
w_bar = 0.4
ax.bar(x - w_bar/2, shs_9c, w_bar, color='steelblue', alpha=0.85,
       label=f'9-C (28 tickers)  mean {stats_9c["mean_sharpe"]:+.2f}')
ax.bar(x + w_bar/2, shs_10a, w_bar, color='darkorange', alpha=0.85,
       label=f'10-A (99 tickers) mean {stats_10a["mean_sharpe"]:+.2f}')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'W{w["idx"]}\n{w["test_range"][0].year}-{w["test_range"][0].month:02d}'
                    for w in windows], fontsize=8)
ax.set_xlabel('Walk-forward window (test start)')
ax.set_ylabel('Sharpe ratio (out-of-sample test)')
ax.set_title('Walk-forward Sharpe — 28 tickers (9-C) vs 99 tickers (10-A)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# (2, 1) Sharpe boxplot
ax = fig.add_subplot(gs[1, 0])
bp = ax.boxplot([shs_9c, shs_10a],
                labels=['9-C (28)', '10-A (99)'],
                patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['steelblue', 'darkorange']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color='black', lw=0.5, ls='--')
for i, (s, label) in enumerate([(stats_9c, 'mean'), (stats_10a, 'mean')]):
    ax.scatter([i + 1], [s['mean_sharpe']], color='red', marker='D', s=80,
               zorder=3, label=label if i == 0 else None)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Sharpe distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (2, 2) Mean Sharpe with CI
ax = fig.add_subplot(gs[1, 1])
labels = ['9-C\n(28 tickers)', '10-A\n(99 tickers)']
means = [stats_9c['mean_sharpe'], stats_10a['mean_sharpe']]
los = [stats_9c['ci'][0], stats_10a['ci'][0]]
his = [stats_9c['ci'][1], stats_10a['ci'][1]]
errs_lo = [m - lo for m, lo in zip(means, los)]
errs_hi = [hi - m for m, hi in zip(means, his)]
ax.errorbar(range(2), means, yerr=[errs_lo, errs_hi],
            fmt='o', capsize=12, capthick=2, lw=2, markersize=14,
            color='black', ecolor='gray')
for i, color in enumerate(['steelblue', 'darkorange']):
    ax.scatter([i], [means[i]], color=color, s=250, zorder=3,
               edgecolor='black', lw=1.5)
ax.axhline(0, color='red', lw=1, ls='--', alpha=0.7, label='Zero alpha')
ax.set_xticks(range(2))
ax.set_xticklabels(labels)
ax.set_ylabel('Mean Sharpe with 95% CI')
ax.set_title('Mean Sharpe + bootstrap 95% CI')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (3, 1) 양수 비율
ax = fig.add_subplot(gs[2, 0])
pos_pcts = [stats_9c['positive_pct'], stats_10a['positive_pct']]
bars = ax.bar(labels, pos_pcts, color=['steelblue', 'darkorange'], alpha=0.85)
ax.axhline(50, color='red', ls='--', alpha=0.7, label='50% (random)')
ax.axhline(60, color='orange', ls=':', alpha=0.7, label='60% threshold')
for bar, val in zip(bars, pos_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.set_ylim(0, 100)
ax.set_ylabel('% of windows with positive Sharpe')
ax.set_title('Positive-Sharpe window ratio')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 2) 풀 크기 추이
ax = fig.add_subplot(gs[2, 1])
ax.plot(x, sizes_9c, 'o-', color='steelblue', lw=2, markersize=8,
        label='9-C (28 tickers) used')
ax.plot(x, sizes_10a, 's-', color='darkorange', lw=2, markersize=8,
        label='10-A (99 tickers) used')
ax.plot(x, fulls_10a, '^--', color='darkorange', alpha=0.4, lw=1.5, markersize=8,
        label='10-A full (before cap)')
ax.set_xticks(x)
ax.set_xticklabels([f'W{w["idx"]}' for w in windows], fontsize=8)
ax.set_xlabel('Window')
ax.set_ylabel('Pair pool size')
ax.set_title('Pair pool size — discovery rate')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('MVP 10-A: Expanded universe (99) vs original (28) — walk-forward',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp10a_universe_expansion.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 결론
# ============================================================
print()
print('=' * 80)
print('결론')
print('=' * 80)
print()


def verdict(stats):
    lo, hi = stats['ci']
    pos = stats['positive_pct']
    if lo > 0:
        return '✓✓ 진짜 alpha (95% CI 양수)'
    elif pos >= 60 and lo > -0.2:
        return '△ 약한 alpha 가능성'
    elif lo <= 0 <= hi:
        return '? 결론 불가 (CI에 0 포함)'
    else:
        return '✗ 음수 alpha (CI 모두 음수)'


print(f'  9-C  (28 tickers):  {verdict(stats_9c)}')
print(f'  10-A (99 tickers): {verdict(stats_10a)}')
print()

# 단순 평가
improved = stats_10a['mean_sharpe'] > stats_9c['mean_sharpe']
pos_improved = stats_10a['positive_pct'] > stats_9c['positive_pct']
ci_improved = stats_10a['ci'][0] > stats_9c['ci'][0]

if improved and ci_improved and stats_10a['ci'][0] > 0:
    print('  ★★ Universe 확장 효과 명확 — alpha 발견')
elif improved and pos_improved:
    print('  ★ Universe 확장이 일부 도움 — but 통계적 alpha 아직 부족')
elif not improved:
    print('  ✗ Universe 확장이 도움 안 됨 — 페어 트레이딩 본질적 한계')
else:
    print('  △ 혼재된 결과')

# Save
with open('/tmp/mvp10a_summary.json', 'w') as f:
    json.dump({
        '9c': stats_9c,
        '10a': stats_10a,
        'delta_mean_sharpe': stats_10a['mean_sharpe'] - stats_9c['mean_sharpe'],
        'delta_positive_pct': stats_10a['positive_pct'] - stats_9c['positive_pct'],
    }, f, indent=2)
print()
print('Summary 저장: /tmp/mvp10a_summary.json')
