"""
6차 MVP: 롤링 페어 재발굴 — 시간 따라 페어 풀이 어떻게 변하는가
================================================================

핵심 질문:
  1. GDX~KO는 언제 페어 풀에서 빠지는가? (regime change 자동 감지)
  2. 새 페어가 그 자리를 채우는가?
  3. 영구히 살아남는 "진짜 견고한" 페어가 있는가?
  4. 1~5차 MVP가 놓친 페어들이 다른 시점엔 살아있나?

핵심 측정:
  - 페어별 생애주기 (언제 적격이었나)
  - 신규 영입 / 폐기 / 유지 비율
  - 전체 페어 풀 크기 추이
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder
from rolling_pairs import walk_forward_pair_lifecycle


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


# ============================================================
# 2. Walk-Forward 페어 풀 생애주기 추적
# ============================================================
print()
print('=' * 70)
print('2. 롤링 페어 재발굴 실행')
print('=' * 70)

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                    max_halflife=30.0, min_halflife=1.0)

LOOKBACK_DAYS = 365      # 1년치 데이터로 페어 검정
REFRESH_EVERY = 30       # 30일마다 재검정

print(f'  Lookback: {LOOKBACK_DAYS}일, Refresh: {REFRESH_EVERY}일마다')
print()

result = walk_forward_pair_lifecycle(
    data, finder,
    lookback_days=LOOKBACK_DAYS,
    refresh_every_days=REFRESH_EVERY,
    verbose=True,
)


# ============================================================
# 3. 요약 통계
# ============================================================
print()
print('=' * 70)
print('3. 요약 통계')
print('=' * 70)
stats = result.summary_stats
print(f'  검정 횟수            : {stats["n_snapshots"]}회')
print(f'  적어도 한 번 적격     : {stats["ever_qualified_pairs"]}개 페어')
print(f'  항상 적격 (영구 견고): {stats["always_qualified"]}개')
print(f'  한 번만 적격 (flash) : {stats["only_once"]}개')
print(f'  평균 풀 크기          : {stats["avg_pool_size"]:.1f}개')
print(f'  평균 신규 영입 / 회   : {stats["avg_new_per_refresh"]:.1f}개')
print(f'  평균 폐기 / 회        : {stats["avg_dropped_per_refresh"]:.1f}개')


# ============================================================
# 4. 페어별 생애주기 (Top 10)
# ============================================================
print()
print('=' * 70)
print('4. 페어별 생애주기 — 누가 오래 살아남았나')
print('=' * 70)
print()
print('  순위  페어              적격 일수    적격 비율')
top_pairs = result.pair_durations.head(15)
total_period = (data.index[-1] - data.index[0]).days
for i, (pair_id, days) in enumerate(top_pairs.items(), 1):
    pct = days / total_period
    print(f'  {i:>3d}.  {pair_id:<18s} {days:>4d}일    {pct:>6.1%}')


# ============================================================
# 5. GDX~KO 추적 — 핵심 관심사
# ============================================================
print()
print('=' * 70)
print('5. GDX~KO 추적 — 언제 페어 풀에서 빠지는가?')
print('=' * 70)
lifecycle = result.pair_lifecycle

# GDX~KO와 GDX~GDXJ 확인
target_pairs = ['GDX~KO', 'KO~GDX', 'GDX~GDXJ', 'GDXJ~GDX']
print()
for tp in target_pairs:
    if tp in lifecycle.columns:
        series = lifecycle[tp]
        active_dates = series[series].index
        if len(active_dates) > 0:
            print(f'  {tp}:')
            print(f'    첫 적격: {active_dates[0].date()}')
            print(f'    마지막 적격: {active_dates[-1].date()}')
            print(f'    총 적격 횟수: {len(active_dates)} / {len(series)}')
            # 끊긴 구간 찾기
            if not series.all():
                gaps = []
                in_gap = False
                gap_start = None
                for d in series.index:
                    if not series[d] and not in_gap:
                        in_gap = True
                        gap_start = d
                    elif series[d] and in_gap:
                        gaps.append((gap_start, d))
                        in_gap = False
                if gaps:
                    print(f'    중단 구간: {len(gaps)}회')
                    for gs, ge in gaps[:3]:
                        print(f'      {gs.date()} ~ {ge.date()}')


# ============================================================
# 6. 시각화
# ============================================================
print()
print('=' * 70)
print('6. 시각화')
print('=' * 70)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)

# (1,1) Gantt-style: 페어별 적격 구간
ax = fig.add_subplot(gs[0, :])  # 위쪽 행 전체
# 적격 횟수 많은 순으로 정렬 (위쪽이 오래 산 페어)
pair_order = result.pair_durations.head(20).index.tolist()  # Top 20
lifecycle_subset = lifecycle[pair_order]

# 각 페어의 시점별 boolean을 시각화
for i, pair_id in enumerate(pair_order):
    series = lifecycle_subset[pair_id]
    for date_idx, date in enumerate(series.index):
        if series[date]:
            # 적격이면 가로 막대 그리기
            color = 'darkgreen'
            if 'GDX' in pair_id and 'KO' in pair_id:
                color = 'red'  # GDX~KO 강조
            ax.barh(i, REFRESH_EVERY, left=date,
                    height=0.7, color=color, alpha=0.7)

ax.set_yticks(range(len(pair_order)))
ax.set_yticklabels(pair_order, fontsize=9)
ax.invert_yaxis()  # 위에서 아래로
ax.set_xlabel('Date')
ax.set_title(f'Pair qualification timeline (Top 20, green=qualified, red=GDX~KO)',
             fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.2, axis='x')

# (2,1) 페어 풀 크기 추이
ax = fig.add_subplot(gs[1, 0])
dates = [s.date for s in result.snapshots]
pool_sizes = [s.n_pairs for s in result.snapshots]
ax.plot(dates, pool_sizes, 'o-', color='steelblue', lw=2, markersize=6)
ax.fill_between(dates, 0, pool_sizes, alpha=0.2, color='steelblue')
ax.set_ylabel('Pool size')
ax.set_xlabel('Date')
ax.set_title('Total pool size over time')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)

# (2,2) 신규 영입 / 폐기 / 유지 추이
ax = fig.add_subplot(gs[1, 1])
n_new = [len(s.new_pairs) for s in result.snapshots[1:]]
n_dropped = [len(s.dropped_pairs) for s in result.snapshots[1:]]
n_survived = [len(s.survived_pairs) for s in result.snapshots[1:]]
x_dates = [s.date for s in result.snapshots[1:]]
width = pd.Timedelta(days=REFRESH_EVERY * 0.3)
ax.bar([d - width for d in x_dates], n_survived, width=width,
       color='steelblue', alpha=0.7, label='Survived')
ax.bar(x_dates, n_new, width=width,
       color='darkgreen', alpha=0.7, label='New')
ax.bar([d + width for d in x_dates], n_dropped, width=width,
       color='red', alpha=0.7, label='Dropped')
ax.set_ylabel('Pair count')
ax.set_xlabel('Date')
ax.set_title('Pool turnover — new / survived / dropped')
ax.legend(loc='upper right', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3, axis='y')

# (3,1) 페어별 총 적격 일수 막대 (top 15)
ax = fig.add_subplot(gs[2, 0])
top15 = result.pair_durations.head(15)
colors = ['red' if 'GDX' in p and 'KO' in p else 'steelblue'
          for p in top15.index]
bars = ax.barh(range(len(top15)), top15.values, color=colors, alpha=0.7)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total qualified days')
ax.set_title('Longest-lived pairs (Top 15)')
ax.grid(alpha=0.3, axis='x')

# (3,2) 1차 MVP에 잡힌 페어들이 6차 MVP 시간 동안 어떻게 보였나
ax = fig.add_subplot(gs[2, 1])
# 1차 MVP 페어: GDX~KO, GDXJ~GLD, BAC~NVDA, NVDA~VZ, MA~WFC, MA~T, JPM~SLV
mvp1_pairs = ['GDX~KO', 'GDXJ~GLD', 'BAC~NVDA', 'NVDA~VZ',
              'MA~WFC', 'MA~T', 'JPM~SLV']
mvp1_durations = []
for p in mvp1_pairs:
    rev = '~'.join(reversed(p.split('~')))
    days = 0
    if p in result.pair_durations.index:
        days = result.pair_durations[p]
    elif rev in result.pair_durations.index:
        days = result.pair_durations[rev]
    mvp1_durations.append(days)

colors2 = ['red' if 'GDX' in p and 'KO' in p else
           'darkgreen' if days > 200 else 'orange' if days > 0 else 'gray'
           for p, days in zip(mvp1_pairs, mvp1_durations)]
ax.barh(range(len(mvp1_pairs)), mvp1_durations, color=colors2, alpha=0.7)
ax.set_yticks(range(len(mvp1_pairs)))
ax.set_yticklabels(mvp1_pairs, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total qualified days')
ax.set_title('MVP1 pairs: how long did they survive in MVP6?')
ax.grid(alpha=0.3, axis='x')

plt.suptitle(f'Rolling Pair Re-discovery '
             f'(Lookback={LOOKBACK_DAYS}d, Refresh={REFRESH_EVERY}d)',
             fontsize=13, y=0.995)

out_path = '/home/claude/rolling_pairs_chart.png'
plt.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close()
print(f'  차트 저장: {out_path}')


# ============================================================
# 7. 핵심 결론
# ============================================================
print()
print('=' * 70)
print('7. 핵심 결론')
print('=' * 70)
print()

# GDX~KO 분석
gdx_ko_in = 'GDX~KO' in lifecycle.columns
ko_gdx_in = 'KO~GDX' in lifecycle.columns

if gdx_ko_in:
    series = lifecycle['GDX~KO']
    n_active = series.sum()
    last_active = series[series].index[-1] if n_active > 0 else None
    print(f'  ◇ GDX~KO: {n_active}/{len(series)}회 적격')
    if last_active and last_active < lifecycle.index[-1]:
        print(f'    → 마지막 적격: {last_active.date()}')
        print(f'    → 그 이후 페어 풀에서 자동 탈락 (regime change 자동 감지!)')

# 1차 MVP의 단일 split 비교
print()
print('  1차 MVP가 단일 split으로 잡았던 7개 페어 중:')
for p, days in zip(mvp1_pairs, mvp1_durations):
    if days == 0:
        print(f'    ✗ {p}: 6차에선 한 번도 적격 안 됨')
    elif days < 100:
        print(f'    △ {p}: {days}일만 적격 (불안정)')
    else:
        print(f'    ✓ {p}: {days}일 적격 (안정)')

print()
print('  핵심 메시지:')
print('    "페어 풀을 한 번 결정하면 시간이 지나면서 죽는다."')
print(f'    회전율: 평균 {stats["avg_new_per_refresh"]:.1f}개 신규 / '
      f'{stats["avg_dropped_per_refresh"]:.1f}개 폐기 매 {REFRESH_EVERY}일')
print('    → 롤링 재발굴 없이는 1~5차 MVP의 모든 노력이 시간이 지나며 무력화됨')
