"""
9차 MVP-C: Walk-Forward 검증
================================

9-E 메타 교훈("단일 백테스트는 신뢰 불가") 직접 후속.

설계:
  Train 2년(504거래일) + Test 1년(252거래일), Step 6개월(126거래일)
  10년 데이터 → 약 12-14개 윈도우
  3 변형 × 윈도우 = 36 백테스트, 분할 실행

평가:
  각 윈도우 test 기간의 Sharpe/Return/MDD 분포
  - 평균, 중앙값, 표준편차
  - 양수 Sharpe 윈도우 비율 (>60%면 진짜 alpha)
  - 부트스트랩 95% 신뢰구간

STEP 1: 윈도우 정의 + (b) 변형 백테스트 (가장 빠름)
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, Backtester, compute_spread,
)

# 9-E 캐시에서 데이터 재활용
CACHE_9E = '/tmp/mvp9e_cache.pkl'
CACHE = '/tmp/mvp9c_cache.pkl'

if os.path.exists(CACHE):
    cache = pickle.load(open(CACHE, 'rb'))
    print(f'기존 캐시 load: keys={list(cache.keys())}')
else:
    cache = {}
    # 9-E 캐시에서 데이터 복사
    if not os.path.exists(CACHE_9E):
        print('ERROR: 9-E 캐시 없음. 데이터 새로 다운로드 필요')
        sys.exit(1)
    cache_9e = pickle.load(open(CACHE_9E, 'rb'))
    cache['data'] = cache_9e['data']
    pickle.dump(cache, open(CACHE, 'wb'))
    print('새 캐시 생성 with 9-E data')

data = cache['data']
print(f'데이터: {data.shape}, {data.index[0].date()} ~ {data.index[-1].date()}')


# ============================================================
# 윈도우 정의
# ============================================================
TRAIN_DAYS = 504    # 거래일 ≈ 2년
TEST_DAYS = 252     # 거래일 ≈ 1년
STEP_DAYS = 126     # 거래일 ≈ 6개월

windows = []
i = 0
while i + TRAIN_DAYS + TEST_DAYS <= len(data):
    train_start = i
    train_end = i + TRAIN_DAYS
    test_start = train_end
    test_end = train_end + TEST_DAYS
    windows.append({
        'idx': len(windows),
        'train_range': (data.index[train_start], data.index[train_end - 1]),
        'test_range': (data.index[test_start], data.index[test_end - 1]),
        'train_loc': (train_start, train_end),
        'test_loc': (test_start, test_end),
    })
    i += STEP_DAYS

cache['windows'] = windows
pickle.dump(cache, open(CACHE, 'wb'))

print()
print(f'윈도우 정의: 총 {len(windows)}개')
print(f'  Train: 504 거래일 (~2년)')
print(f'  Test : 252 거래일 (~1년)')
print(f'  Step : 126 거래일 (~6개월)')
print()
for w in windows:
    print(f'  W{w["idx"]:>2d}: train [{w["train_range"][0].date()} ~ {w["train_range"][1].date()}]  '
          f'test [{w["test_range"][0].date()} ~ {w["test_range"][1].date()}]')


# ============================================================
# (b) 변형 walk-forward
# ============================================================
def calc_metrics(eq):
    if len(eq) < 10:
        return 0.0, 0.0, 0.0
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return ret, sharpe, mdd


print()
print('=' * 72)
print('(b) 변형 walk-forward')
print('=' * 72)

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)
INITIAL_CAPITAL = 100_000

for w in windows:
    key = f'b_w{w["idx"]}'
    if key in cache:
        r = cache[key]
        print(f'  W{w["idx"]:>2d} (b) 캐시 hit: '
              f'Ret {r["return"]:+.2%}  Sh {r["sharpe"]:+.2f}  pool {r["pool_size"]}')
        continue

    t0 = time.time()
    # train 데이터로 PairsFinder
    train_data = data.iloc[w['train_loc'][0]:w['train_loc'][1]]
    fixed_pool = finder.screen_pairs(train_data)

    if not fixed_pool:
        cache[key] = {
            'window': w['idx'],
            'pool_size': 0,
            'return': 0.0, 'sharpe': 0.0, 'mdd': 0.0,
            'test_equity': None,
            'pool_pairs': [],
        }
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'  W{w["idx"]:>2d} (b) 페어 발굴 0개 → 현금 보유 [{time.time()-t0:.1f}s]')
        continue

    # test 기간에 페어별 백테스트 → 균등 가중치 합산
    test_data = data.iloc[w['test_loc'][0]:w['test_loc'][1]]
    daily_returns = pd.Series(0.0, index=test_data.index)
    equal_w = 0.10 / len(fixed_pool)

    for p in fixed_pool:
        # spread는 train+test 전체로 계산 (z-score window 필요)
        # but signal/backtest는 test 기간만
        full_data = pd.concat([train_data, test_data])
        spread = compute_spread(full_data[p.y], full_data[p.x], p.beta)
        sw = max(20, min(60, int(p.half_life * 2)))
        sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
        z, pos, fc = sg.generate(spread)
        # test 기간만 잘라서 backtest
        test_pos = pos.loc[test_data.index]
        test_fc = fc.loc[test_data.index] if hasattr(fc, 'loc') else fc
        bt = Backtester(initial_capital=INITIAL_CAPITAL,
                        capital_fraction=1.0,
                        fee_rate=0.0004, slippage=0.0005)
        res = bt.run(test_data[p.y], test_data[p.x], p.beta, test_pos, test_fc)
        daily_returns += equal_w * res.equity.pct_change().fillna(0)

    test_equity = INITIAL_CAPITAL * (1 + daily_returns).cumprod()
    ret, sh, mdd = calc_metrics(test_equity)

    cache[key] = {
        'window': w['idx'],
        'pool_size': len(fixed_pool),
        'return': ret, 'sharpe': sh, 'mdd': mdd,
        'test_equity': test_equity,
        'pool_pairs': [(p.y, p.x) for p in fixed_pool],
    }
    pickle.dump(cache, open(CACHE, 'wb'))

    print(f'  W{w["idx"]:>2d} (b) pool={len(fixed_pool)}  '
          f'Ret {ret:+7.2%}  Sh {sh:+5.2f}  MDD {mdd:+6.2%}  '
          f'[{time.time()-t0:.1f}s]')

print()
print(f'완료. 캐시 키 수: {len(cache)}')
