"""9-C STEP 2: r180_cd30 walk-forward — 9-D Sharpe 1위 변형"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder
from production_system import ProductionSystem

CACHE = '/tmp/mvp9c_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))
data = cache['data']
windows = cache['windows']
print(f'데이터 {data.shape}, 윈도우 {len(windows)}개')


def calc_metrics(eq):
    if len(eq) < 10:
        return 0.0, 0.0, 0.0
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return ret, sharpe, mdd


finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)

COMMON = dict(finder=finder, lookback_days=365,
              sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
              initial_capital=100_000,
              capital_per_pair_cap=0.10,
              portfolio_kelly_fraction=0.25,
              use_history_for_kelly=True,
              max_active_pairs=5,
              min_historical_sharpe=0.0,
              quality_lookback=90)

LABEL = 'r180_cd30'
PARAMS = {'refresh_every_days': 180, 'pair_cooldown_days': 30}

print()
print(f'{LABEL} walk-forward 시작')
print()

for w in windows:
    key = f'{LABEL}_w{w["idx"]}'
    if key in cache:
        r = cache[key]
        print(f'  W{w["idx"]:>2d} {LABEL} 캐시 hit: '
              f'Ret {r["return"]:+.2%}  Sh {r["sharpe"]:+.2f}')
        continue

    t0 = time.time()
    # 윈도우 전체 데이터 (train + test)
    window_data = data.iloc[w['train_loc'][0]:w['test_loc'][1]]
    test_start_date = data.index[w['test_loc'][0]]
    test_end_date = data.index[w['test_loc'][1] - 1]

    sys_v = ProductionSystem(**COMMON, **PARAMS)
    res = sys_v.run(window_data, verbose=False)

    # test 기간만 잘라서 metric 계산
    full_eq = res.equity_curve
    test_eq = full_eq.loc[test_start_date:test_end_date]
    if len(test_eq) < 10:
        ret, sh, mdd = 0.0, 0.0, 0.0
        n_snap = 0
        avg_pairs = 0
    else:
        # test 시작 시점을 $100K로 정규화
        test_eq_norm = test_eq / test_eq.iloc[0] * 100_000
        ret, sh, mdd = calc_metrics(test_eq_norm)

        # test 기간 안에서의 snapshot/페어 통계
        n_snap = sum(1 for s in res.monthly_states
                     if test_start_date <= s.date <= test_end_date)
        active_during_test = [
            len(s.active_pairs) for s in res.monthly_states
            if test_start_date <= s.date <= test_end_date and s.active_pairs
        ]
        avg_pairs = np.mean(active_during_test) if active_during_test else 0

    cache[key] = {
        'window': w['idx'],
        'return': ret, 'sharpe': sh, 'mdd': mdd,
        'test_equity': test_eq_norm if len(test_eq) >= 10 else None,
        'n_test_snapshots': n_snap,
        'avg_active_pairs': float(avg_pairs),
    }
    pickle.dump(cache, open(CACHE, 'wb'))

    print(f'  W{w["idx"]:>2d} {LABEL} '
          f'Ret {ret:+7.2%}  Sh {sh:+5.2f}  MDD {mdd:+6.2%}  '
          f'test_snap {n_snap} pairs {avg_pairs:.1f}  '
          f'[{time.time()-t0:.1f}s]')

print()
print(f'완료. 캐시 키 수: {len(cache)}')
