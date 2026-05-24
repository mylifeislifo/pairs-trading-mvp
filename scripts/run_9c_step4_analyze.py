"""9-C STEP 4: Walk-forward 결과 분석 + 시각화"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp9c_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))
data = cache['data']
windows = cache['windows']

print(f'데이터 {data.shape}, 윈도우 {len(windows)}개')

VARIANTS = ['b', 'r180_cd30', 'r30_cd0']
DISPLAY = {'b': '(b) Fixed pool', 'r180_cd30': 'r180, cd=30', 'r30_cd0': 'r30, cd=0'}
COLORS = {'b': 'darkblue', 'r180_cd30': 'purple', 'r30_cd0': 'red'}


# ============================================================
# 각 변형별 윈도우별 결과 수집
# ============================================================
def collect(variant):
    rets = []
    shs = []
    mdds = []
    for w in windows:
        key = f'{variant}_w{w["idx"]}'
        r = cache.get(key, {})
        rets.append(r.get('return', 0))
        shs.append(r.get('sharpe', 0))
        mdds.append(r.get('mdd', 0))
    return np.array(rets), np.array(shs), np.array(mdds)


results = {v: collect(v) for v in VARIANTS}


# ============================================================
# 통계 요약
# ============================================================
print()
print('=' * 80)
print('Walk-Forward 결과 요약 (윈도우 14개)')
print('=' * 80)
print()
print(f'  {"Variant":<16s} {"평균Ret":>9s} {"중앙Ret":>9s} '
      f'{"평균Sh":>8s} {"중앙Sh":>8s} {"표준편차":>9s} '
      f'{"양수Sh%":>8s} {"평균MDD":>9s}')
print(f'  {"-"*16} {"-"*9} {"-"*9} {"-"*8} {"-"*8} {"-"*9} {"-"*8} {"-"*9}')

summary = {}
for v in VARIANTS:
    rets, shs, mdds = results[v]
    mean_ret = np.mean(rets)
    median_ret = np.median(rets)
    mean_sh = np.mean(shs)
    median_sh = np.median(shs)
    std_sh = np.std(shs)
    pos_pct = (shs > 0).mean() * 100
    mean_mdd = np.mean(mdds)
    summary[v] = {
        'mean_ret': float(mean_ret), 'median_ret': float(median_ret),
        'mean_sharpe': float(mean_sh), 'median_sharpe': float(median_sh),
        'std_sharpe': float(std_sh),
        'positive_pct': float(pos_pct),
        'mean_mdd': float(mean_mdd),
        'all_rets': rets.tolist(),
        'all_shs': shs.tolist(),
        'all_mdds': mdds.tolist(),
    }
    print(f'  {DISPLAY[v]:<16s} '
          f'{mean_ret*100:>+8.2f}% '
          f'{median_ret*100:>+8.2f}% '
          f'{mean_sh:>+8.2f} '
          f'{median_sh:>+8.2f} '
          f'{std_sh:>9.2f} '
          f'{pos_pct:>7.1f}% '
          f'{mean_mdd*100:>+8.2f}%')


# ============================================================
# 부트스트랩 95% CI for mean Sharpe
# ============================================================
print()
print('=' * 80)
print('부트스트랩 95% 신뢰구간 (Sharpe)')
print('=' * 80)
print()

def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    """평균 mean의 부트스트랩 신뢰구간"""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    lo = np.quantile(means, alpha)
    hi = np.quantile(means, 1 - alpha)
    return float(lo), float(hi)


for v in VARIANTS:
    shs = results[v][1]
    lo, hi = bootstrap_ci(shs)
    summary[v]['ci_95_lo'] = lo
    summary[v]['ci_95_hi'] = hi
    contains_zero = lo <= 0 <= hi
    marker = '⚠ 0 포함' if contains_zero else ('✓ 양수' if lo > 0 else '✗ 음수')
    print(f'  {DISPLAY[v]:<16s}  '
          f'평균 {np.mean(shs):+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]  {marker}')


# ============================================================
# 윈도우별 상세 표
# ============================================================
print()
print('=' * 80)
print('윈도우별 Sharpe 상세 (test 기간)')
print('=' * 80)
print()
print(f'  {"W":>3s}  {"test 기간":<23s} '
      f'{DISPLAY["b"]:>14s} {DISPLAY["r180_cd30"]:>14s} {DISPLAY["r30_cd0"]:>14s}')
print(f'  {"-"*3}  {"-"*23} {"-"*14} {"-"*14} {"-"*14}')

for w in windows:
    line = f'  W{w["idx"]:>2d}  [{w["test_range"][0].date()}~{w["test_range"][1].date()}]'
    for v in VARIANTS:
        sh = results[v][1][w['idx']]
        line += f'  {sh:>+12.2f}  '
    print(line)


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

# (1, full row) Sharpe per window — 모든 변형
ax = fig.add_subplot(gs[0, :])
x = np.arange(len(windows))
w_bar = 0.27
for i, v in enumerate(VARIANTS):
    shs = results[v][1]
    ax.bar(x + (i - 1) * w_bar, shs, w_bar,
           color=COLORS[v], alpha=0.85, label=DISPLAY[v])
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'W{w["idx"]}\n{w["test_range"][0].year}-{w["test_range"][0].month:02d}'
                    for w in windows], fontsize=8)
ax.set_xlabel('Walk-forward window (test start)')
ax.set_ylabel('Sharpe ratio (out-of-sample test)')
ax.set_title('Walk-forward Sharpe per window (14 OOS windows × 3 variants)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# (2, 1) Sharpe 분포 boxplot
ax = fig.add_subplot(gs[1, 0])
all_shs = [results[v][1] for v in VARIANTS]
bp = ax.boxplot(all_shs, labels=[DISPLAY[v] for v in VARIANTS],
                patch_artist=True, widths=0.5)
for patch, v in zip(bp['boxes'], VARIANTS):
    patch.set_facecolor(COLORS[v])
    patch.set_alpha(0.7)
ax.axhline(0, color='black', lw=0.5, ls='--')
# 평균 표시
for i, v in enumerate(VARIANTS):
    mean_sh = np.mean(results[v][1])
    ax.scatter([i + 1], [mean_sh], color='red', marker='D', s=80, zorder=3,
               label='Mean' if i == 0 else None)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Sharpe distribution across walk-forward windows')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (2, 2) Mean Sharpe with bootstrap 95% CI
ax = fig.add_subplot(gs[1, 1])
means = [np.mean(results[v][1]) for v in VARIANTS]
los = [summary[v]['ci_95_lo'] for v in VARIANTS]
his = [summary[v]['ci_95_hi'] for v in VARIANTS]
errors_lo = [m - lo for m, lo in zip(means, los)]
errors_hi = [hi - m for m, hi in zip(means, his)]
colors_seq = [COLORS[v] for v in VARIANTS]
ax.errorbar(range(len(VARIANTS)), means, yerr=[errors_lo, errors_hi],
            fmt='o', capsize=10, capthick=2, lw=2, markersize=12,
            color='black', ecolor='gray')
for i, v in enumerate(VARIANTS):
    ax.scatter([i], [means[i]], color=COLORS[v], s=200, zorder=3,
               edgecolor='black', lw=1.5)
ax.axhline(0, color='red', lw=1, ls='--', alpha=0.7, label='Zero alpha')
ax.set_xticks(range(len(VARIANTS)))
ax.set_xticklabels([DISPLAY[v] for v in VARIANTS])
ax.set_ylabel('Mean Sharpe ratio')
ax.set_title('Mean Sharpe with bootstrap 95% CI — does it exclude zero?')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (3, 1) 양수 Sharpe 윈도우 비율
ax = fig.add_subplot(gs[2, 0])
pos_pcts = [summary[v]['positive_pct'] for v in VARIANTS]
bars = ax.bar([DISPLAY[v] for v in VARIANTS], pos_pcts, color=colors_seq, alpha=0.85)
ax.axhline(50, color='red', ls='--', alpha=0.7, label='50% (random)')
ax.axhline(60, color='orange', ls=':', alpha=0.7, label='60% threshold')
for bar, val in zip(bars, pos_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.set_ylabel('% of windows with positive Sharpe')
ax.set_title('Positive-Sharpe window ratio — alpha indicator')
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 2) 윈도우별 시기 + 시장 컨텍스트 주석
ax = fig.add_subplot(gs[2, 1])
# (b) 윈도우별 Sharpe + cumulative
for v in VARIANTS:
    shs = results[v][1]
    ax.plot(x, shs, 'o-', color=COLORS[v], label=DISPLAY[v], lw=1.5, markersize=7)
# 주요 시장 이벤트 영역 표시
events = [
    (2, 4, 'COVID', 'pink'),
    (7, 9, '2022 inflation', 'lightyellow'),
]
for start, end, label, color in events:
    ax.axvspan(start, end, alpha=0.3, color=color, label=label)
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'W{w["idx"]}' for w in windows], fontsize=8)
ax.set_xlabel('Window')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Sharpe across time — market regime context')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)

plt.suptitle('MVP 9-C: Walk-forward validation — distribution of out-of-sample Sharpe',
             fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9c_walkforward.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 핵심 결론
# ============================================================
print()
print('=' * 80)
print('핵심 결론')
print('=' * 80)
print()

# 각 변형별 평결
for v in VARIANTS:
    s = summary[v]
    lo, hi = s['ci_95_lo'], s['ci_95_hi']
    contains_zero = lo <= 0 <= hi
    pos = s['positive_pct']

    if not contains_zero and lo > 0:
        verdict = '✓✓ 진짜 alpha (95% CI 모두 양수)'
    elif pos >= 60 and lo > -0.2:
        verdict = '△ 약한 alpha 가능성 (양수 비율 60% 이상)'
    elif contains_zero:
        verdict = '? 결론 불가 (95% CI에 0 포함)'
    else:
        verdict = '✗ 음수 alpha (95% CI 모두 음수)'

    print(f'  {DISPLAY[v]:<16s}')
    print(f'    평균 Sharpe: {s["mean_sharpe"]:+.2f}, CI [{lo:+.2f}, {hi:+.2f}]')
    print(f'    양수 비율  : {pos:.1f}%')
    print(f'    평결       : {verdict}')
    print()

# 데이터 dump
with open('/tmp/mvp9c_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary 저장: /tmp/mvp9c_summary.json')
