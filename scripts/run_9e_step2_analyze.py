"""
9-E Step 2: 캐시된 결과 분석 + 시각화 + 시간 분할 안정성
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp9e_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))

data = cache['data']
print(f'데이터: {data.shape}, {data.index[0].date()} ~ {data.index[-1].date()}')

INITIAL_CAPITAL = 100_000


def calc_metrics(eq):
    ret = eq.iloc[-1]/eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    mdd = float((eq/eq.cummax()-1).min())
    return ret, sharpe, mdd


def calc_metrics_slice(eq, start_date, end_date):
    slc = eq.loc[start_date:end_date]
    if len(slc) < 10:
        return None, None, None
    ret = slc.iloc[-1]/slc.iloc[0] - 1
    daily = slc.pct_change().dropna()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    mdd = float((slc/slc.cummax()-1).min())
    return ret, sharpe, mdd


# ============================================================
# (b) metrics 보완
# ============================================================
equity_b = cache['b']['equity']
ret_b, sharpe_b, mdd_b = calc_metrics(equity_b)
fixed_pool_size = cache['b']['fixed_pool_size']
print(f'\n(b) 풀: {fixed_pool_size}개 페어 (10년 train에서 발굴)')
for y, x in cache['b']['fixed_pool_pairs']:
    print(f'  {y} ~ {x}')
print(f'(b) Ret {ret_b:+.2%}, Sharpe {sharpe_b:+.2f}, MDD {mdd_b:+.2%}')


# ============================================================
# 변형 결과 정리
# ============================================================
VARIANT_ORDER = ['r30_cd0', 'r30_cd30', 'r90_cd30', 'r180_cd0', 'r180_cd30']
LABELS_DISPLAY = {
    'r30_cd0':   'r30, cd=0',
    'r30_cd30':  'r30, cd=30',
    'r90_cd30':  'r90, cd=30',
    'r180_cd0':  'r180, cd=0',
    'r180_cd30': 'r180, cd=30',
}

# 9-D 3년 결과 (수치는 9-D 보고서에서 가져옴)
RESULTS_3YR = {
    'r30_cd0':   {'ret': -0.0653, 'sharpe': -0.45, 'mdd': -0.1422},
    'r30_cd30':  {'ret': +0.0023, 'sharpe': +0.04, 'mdd': -0.0686},
    'r90_cd30':  {'ret': +0.0267, 'sharpe': +0.48, 'mdd': -0.0297},
    'r180_cd0':  {'ret': +0.0654, 'sharpe': +0.32, 'mdd': -0.0742},
    'r180_cd30': {'ret': +0.0293, 'sharpe': +0.73, 'mdd': -0.0121},
}
B_3YR = {'ret': +0.0294, 'sharpe': +0.83, 'mdd': -0.0220}


# ============================================================
# 종합 테이블
# ============================================================
print()
print('=' * 80)
print('10년 종합 결과')
print('=' * 80)
print()
print(f'  {"Variant":<13s} {"최종":>11s} {"Return":>9s} {"Sharpe":>7s} '
      f'{"MDD":>8s} {"snap":>5s} {"pairs":>6s} {"new":>5s}')
print(f'  {"-"*13} {"-"*11} {"-"*9} {"-"*7} {"-"*8} {"-"*5} {"-"*6} {"-"*5}')
for label in VARIANT_ORDER:
    r = cache[label]
    print(f'  {LABELS_DISPLAY[label]:<13s} '
          f'${r["equity"].iloc[-1]:>9,.0f}  '
          f'{r["return"]:>+8.2%} '
          f'{r["sharpe"]:>+7.2f} '
          f'{r["mdd"]:>+7.2%} '
          f'{r["n_snapshots"]:>5d} '
          f'{r["avg_pairs"]:>6.1f} '
          f'{r["n_new"]:>5d}')
print(f'  {"-"*13}')
print(f'  {"(b) ref":<13s} '
      f'${equity_b.iloc[-1]:>9,.0f}  '
      f'{ret_b:>+8.2%} '
      f'{sharpe_b:>+7.2f} '
      f'{mdd_b:>+7.2%} '
      f'{"-":>5s} '
      f'{fixed_pool_size:>6d} '
      f'{"-":>5s}')


# ============================================================
# 3년 vs 10년 견고성 판정
# ============================================================
print()
print('=' * 80)
print('3년 vs 10년 견고성 — 핵심 견정')
print('=' * 80)
print()
print(f'  {"Variant":<13s} {"3년Ret":>9s} {"10년Ret":>10s} {"3년Sh":>8s} {"10년Sh":>9s} {"판정":>10s}')
print(f'  {"-"*13} {"-"*9} {"-"*10} {"-"*8} {"-"*9} {"-"*10}')
flip_count = 0
for label in VARIANT_ORDER:
    r3 = RESULTS_3YR[label]
    r10 = cache[label]
    sign_match = (r3['sharpe'] > 0) == (r10['sharpe'] > 0)
    if not sign_match:
        flip_count += 1
    verdict = '✓ 견고' if sign_match else '✗ 무너짐'
    print(f'  {LABELS_DISPLAY[label]:<13s} '
          f'{r3["ret"]*100:>+8.2f}% '
          f'{r10["return"]*100:>+9.2f}% '
          f'{r3["sharpe"]:>+7.2f} '
          f'{r10["sharpe"]:>+8.2f} '
          f'{verdict:>10s}')

sign_match_b = (B_3YR['sharpe'] > 0) == (sharpe_b > 0)
print(f'  {"(b) ref":<13s} '
      f'{B_3YR["ret"]*100:>+8.2f}% '
      f'{ret_b*100:>+9.2f}% '
      f'{B_3YR["sharpe"]:>+7.2f} '
      f'{sharpe_b:>+8.2f} '
      f'{"✓ 견고" if sign_match_b else "✗ 무너짐":>10s}')

print()
print(f'  변형 중 부호 반전: {flip_count}/{len(VARIANT_ORDER)}개')


# ============================================================
# 시간 분할 안정성 — 전반/후반 5년
# ============================================================
print()
print('=' * 80)
print('시간 분할 안정성 — 전반 5년 / 후반 5년')
print('=' * 80)

start_date = data.index[0]
mid_date = data.index[len(data) // 2]
end_date = data.index[-1]
print(f'  전반: {start_date.date()} ~ {mid_date.date()}')
print(f'  후반: {mid_date.date()} ~ {end_date.date()}')
print()
print(f'  {"Variant":<13s} {"전반Ret":>9s} {"후반Ret":>9s} {"전반Sh":>8s} {"후반Sh":>8s} {"안정성":>10s}')
print(f'  {"-"*13} {"-"*9} {"-"*9} {"-"*8} {"-"*8} {"-"*10}')

stability_data = {}
for label in VARIANT_ORDER:
    eq = cache[label]['equity']
    r1, s1, m1 = calc_metrics_slice(eq, start_date, mid_date)
    r2, s2, m2 = calc_metrics_slice(eq, mid_date, end_date)
    sign_match = (s1 > 0) == (s2 > 0) if s1 is not None and s2 is not None else False
    stability_data[label] = {'first': (r1, s1, m1), 'second': (r2, s2, m2), 'stable': sign_match}
    print(f'  {LABELS_DISPLAY[label]:<13s} '
          f'{r1*100:>+8.2f}% '
          f'{r2*100:>+8.2f}% '
          f'{s1:>+7.2f} '
          f'{s2:>+7.2f} '
          f'{"✓ 안정" if sign_match else "✗ 불안정":>10s}')

r1b, s1b, _ = calc_metrics_slice(equity_b, start_date, mid_date)
r2b, s2b, _ = calc_metrics_slice(equity_b, mid_date, end_date)
sb_match = (s1b > 0) == (s2b > 0)
stability_data['(b)'] = {'first': (r1b, s1b, _), 'second': (r2b, s2b, _), 'stable': sb_match}
print(f'  {"(b) ref":<13s} '
      f'{r1b*100:>+8.2f}% '
      f'{r2b*100:>+8.2f}% '
      f'{s1b:>+7.2f} '
      f'{s2b:>+7.2f} '
      f'{"✓ 안정" if sb_match else "✗ 불안정":>10s}')


# ============================================================
# 시각화
# ============================================================
print()
print('시각화 생성 중...')

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

color_map = {
    'r30_cd0':   'red',
    'r30_cd30':  'orange',
    'r90_cd30':  'darkgreen',
    'r180_cd0':  'darkorange',
    'r180_cd30': 'purple',
}

# (1, full) Equity curves (log scale)
ax = fig.add_subplot(gs[0, :])
for label in VARIANT_ORDER:
    r = cache[label]
    ax.plot(r['equity'].index, r['equity'],
            label=f'{LABELS_DISPLAY[label]} → {r["return"]:+.2%} (Sh {r["sharpe"]:+.2f})',
            color=color_map[label], lw=1.4)
ax.plot(equity_b.index, equity_b,
        label=f'(b) ref → {ret_b:+.2%} (Sh {sharpe_b:+.2f})',
        color='black', lw=2.2, ls='--')
ax.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.5)
ax.axvline(mid_date, color='red', ls=':', alpha=0.4)
ax.set_ylabel('Equity (USD)')
ax.set_xlabel('Date')
ax.set_title(f'MVP 9-E: 10-year equity curves '
             f'({start_date.date()} ~ {end_date.date()})')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

# (2, 1) 3년 vs 10년 Sharpe
ax = fig.add_subplot(gs[1, 0])
labels_disp = [LABELS_DISPLAY[l] for l in VARIANT_ORDER] + ['(b)']
sh3 = [RESULTS_3YR[l]['sharpe'] for l in VARIANT_ORDER] + [B_3YR['sharpe']]
sh10 = [cache[l]['sharpe'] for l in VARIANT_ORDER] + [sharpe_b]
x = np.arange(len(labels_disp))
w = 0.38
ax.bar(x - w/2, sh3, w, color='steelblue', alpha=0.85, label='3-year (9-D)')
ax.bar(x + w/2, sh10, w, color='darkorange', alpha=0.85, label='10-year (9-E)')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_disp, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Sharpe: 3-year vs 10-year — robustness check')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (2, 2) 3년 vs 10년 Return
ax = fig.add_subplot(gs[1, 1])
ret3 = [RESULTS_3YR[l]['ret']*100 for l in VARIANT_ORDER] + [B_3YR['ret']*100]
ret10 = [cache[l]['return']*100 for l in VARIANT_ORDER] + [ret_b*100]
ax.bar(x - w/2, ret3, w, color='steelblue', alpha=0.85, label='3-year (9-D)')
ax.bar(x + w/2, ret10, w, color='darkorange', alpha=0.85, label='10-year (9-E)')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_disp, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Total Return (%)')
ax.set_title('Total Return: 3-year vs 10-year — note the sign flips')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 1) 전반/후반 Sharpe
ax = fig.add_subplot(gs[2, 0])
sh_first = [stability_data[l]['first'][1] for l in VARIANT_ORDER] + [stability_data['(b)']['first'][1]]
sh_second = [stability_data[l]['second'][1] for l in VARIANT_ORDER] + [stability_data['(b)']['second'][1]]
ax.bar(x - w/2, sh_first, w, color='lightblue', alpha=0.85, label='First half (5y)')
ax.bar(x + w/2, sh_second, w, color='lightcoral', alpha=0.85, label='Second half (5y)')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_disp, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Sharpe ratio')
ax.set_title('Time-split stability — within 10-year period')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (3, 2) Drawdown curves
ax = fig.add_subplot(gs[2, 1])
for label in VARIANT_ORDER:
    r = cache[label]
    dd = (r['equity'] / r['equity'].cummax() - 1) * 100
    ax.plot(dd.index, dd, label=LABELS_DISPLAY[label],
            color=color_map[label], lw=1.2, alpha=0.8)
dd_b = (equity_b / equity_b.cummax() - 1) * 100
ax.plot(dd_b.index, dd_b, color='black', lw=2, ls='--', label='(b) ref')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.set_title('Long-term drawdown')
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)

plt.suptitle('MVP 9-E: 10-year validation — 9-D findings collapsed', fontsize=13, y=0.995)

charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(charts_dir, 'mvp9e_long_horizon.png'))
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'차트 저장: {out_path}')


# ============================================================
# 결론 + 데이터 dump
# ============================================================
print()
print('=' * 80)
print('결론')
print('=' * 80)
print()

if flip_count >= 3:
    print(f'  ★★★ 9-D 결과 완전 인공물: {flip_count}/{len(VARIANT_ORDER)}개 변형 부호 반전')
elif flip_count >= 1:
    print(f'  ★ 9-D 결과 부분 인공물: {flip_count}/{len(VARIANT_ORDER)}개 변형 부호 반전')
else:
    print(f'  ✓ 9-D 결과 견고: 0개 부호 반전')

# 모든 변형이 10년에서 음수인가?
all_negative = all(cache[l]['return'] < 0 for l in VARIANT_ORDER)
b_negative = ret_b < 0
print(f'  10년에서 모든 변형 수익 부호: '
      f'{"모두 음수" if all_negative else "혼합"}')
print(f'  10년에서 (b) 수익 부호: '
      f'{"음수" if b_negative else "양수"}')

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
        'fixed_pool_size': fixed_pool_size,
        'pairs': cache['b']['fixed_pool_pairs'],
    },
    'variants_10yr': {
        label: {
            'return': cache[label]['return'],
            'sharpe': cache[label]['sharpe'],
            'mdd': cache[label]['mdd'],
            'n_snapshots': cache[label]['n_snapshots'],
            'avg_pairs': cache[label]['avg_pairs'],
            'n_new': cache[label]['n_new'],
        } for label in VARIANT_ORDER
    },
    'comparison_3yr_vs_10yr': {
        label: {
            '3yr_sharpe': RESULTS_3YR[label]['sharpe'],
            '10yr_sharpe': cache[label]['sharpe'],
            '3yr_ret': RESULTS_3YR[label]['ret'],
            '10yr_ret': cache[label]['return'],
            'sign_preserved': (RESULTS_3YR[label]['sharpe'] > 0) == (cache[label]['sharpe'] > 0),
        } for label in VARIANT_ORDER
    },
    'stability': {
        label: {
            'first_half_sharpe': float(stability_data[label]['first'][1] if stability_data[label]['first'][1] is not None else 0),
            'second_half_sharpe': float(stability_data[label]['second'][1] if stability_data[label]['second'][1] is not None else 0),
            'stable': bool(stability_data[label]['stable']),
        } for label in list(VARIANT_ORDER) + ['(b)']
    },
    'flip_count': flip_count,
    'verdict': 'COMPLETE_ARTIFACT' if flip_count >= 3 else ('PARTIAL_ARTIFACT' if flip_count >= 1 else 'ROBUST'),
}
with open('/tmp/mvp9e_report_data.json', 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
print()
print('데이터 저장: /tmp/mvp9e_report_data.json')
