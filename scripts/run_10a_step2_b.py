"""
10차 MVP-A: STEP 2 — (b) 변형 walk-forward on expanded universe (99 ticker)
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder, SignalGenerator, Backtester, compute_spread

CACHE = '/tmp/mvp10a_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))
data = cache['data']
print(f'데이터: {data.shape}, 종목: {data.shape[1]}')


# ============================================================
# 윈도우 정의 (9-C 동일)
# ============================================================
TRAIN_DAYS = 504
TEST_DAYS = 252
STEP_DAYS = 126

windows = []
i = 0
while i + TRAIN_DAYS + TEST_DAYS <= len(data):
    windows.append({
        'idx': len(windows),
        'train_range': (data.index[i], data.index[i + TRAIN_DAYS - 1]),
        'test_range': (data.index[i + TRAIN_DAYS], data.index[i + TRAIN_DAYS + TEST_DAYS - 1]),
        'train_loc': (i, i + TRAIN_DAYS),
        'test_loc': (i + TRAIN_DAYS, i + TRAIN_DAYS + TEST_DAYS),
    })
    i += STEP_DAYS

cache['windows'] = windows
pickle.dump(cache, open(CACHE, 'wb'))
print(f'윈도우: {len(windows)}개')


def calc_metrics(eq):
    if len(eq) < 10:
        return 0.0, 0.0, 0.0
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    return ret, sharpe, mdd


# ============================================================
# (b) walk-forward
# ============================================================
# 더 엄격한 PairsFinder
finder = PairsFinder(
    alpha_adf=0.05,        # 0.10 → 0.05 (더 엄격)
    pvalue_coint=0.01,     # 0.05 → 0.01 (더 엄격)
    max_halflife=30.0,
    min_halflife=1.0,
)
INITIAL_CAPITAL = 100_000
MAX_PAIRS = 20  # 상위 20개로 cap (페어 수가 폭증할 수 있으므로)

print()
print('=' * 72)
print('(b) walk-forward — expanded universe')
print(f'PairsFinder: alpha_adf=0.05, pvalue_coint=0.01 (엄격)')
print(f'운용 페어: 최대 {MAX_PAIRS}개 (Sharpe 정렬 후 cap)')
print('=' * 72)

for w in windows:
    key = f'b_w{w["idx"]}'
    if key in cache:
        r = cache[key]
        print(f'  W{w["idx"]:>2d} 캐시 hit: Ret {r["return"]:+.2%} Sh {r["sharpe"]:+.2f}')
        continue

    t0 = time.time()
    train_data = data.iloc[w['train_loc'][0]:w['train_loc'][1]]

    # 페어 발굴
    t_screen = time.time()
    fixed_pool_all = finder.screen_pairs(train_data)
    screen_time = time.time() - t_screen
    n_found = len(fixed_pool_all)

    if n_found == 0:
        cache[key] = {
            'window': w['idx'], 'pool_size': 0, 'pool_size_full': 0,
            'return': 0.0, 'sharpe': 0.0, 'mdd': 0.0,
            'test_equity': None, 'pool_pairs': [],
            'screen_time': screen_time,
        }
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'  W{w["idx"]:>2d} 페어 0개 발굴 [{time.time()-t0:.1f}s]')
        continue

    # 상위 N개로 cap — half_life 짧은 순으로 선택 (평균 회귀 빠른 페어 선호)
    # 또는 train의 in-sample Sharpe로 선택
    # 간단하게 half_life 짧은 순 (PairsFinder가 이미 정렬했을 수 있지만 확실히)
    fixed_pool = sorted(fixed_pool_all, key=lambda p: p.half_life)[:MAX_PAIRS]
    n_used = len(fixed_pool)

    # test 기간 백테스트 (균등 가중)
    test_data = data.iloc[w['test_loc'][0]:w['test_loc'][1]]
    daily_returns = pd.Series(0.0, index=test_data.index)
    equal_w = 0.10 / n_used  # 9-C와 동일한 자본 사용 (cap 10% 분산)

    for p in fixed_pool:
        full_data = pd.concat([train_data, test_data])
        try:
            spread = compute_spread(full_data[p.y], full_data[p.x], p.beta)
            sw = max(20, min(60, int(p.half_life * 2)))
            sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
            z, pos, fc = sg.generate(spread)
            test_pos = pos.loc[test_data.index]
            test_fc = fc.loc[test_data.index] if hasattr(fc, 'loc') else fc
            bt = Backtester(initial_capital=INITIAL_CAPITAL,
                            capital_fraction=1.0,
                            fee_rate=0.0004, slippage=0.0005)
            res = bt.run(test_data[p.y], test_data[p.x], p.beta, test_pos, test_fc)
            daily_returns += equal_w * res.equity.pct_change().fillna(0)
        except Exception as e:
            pass  # 개별 페어 오류는 무시

    test_equity = INITIAL_CAPITAL * (1 + daily_returns).cumprod()
    ret, sh, mdd = calc_metrics(test_equity)

    cache[key] = {
        'window': w['idx'],
        'pool_size': n_used,
        'pool_size_full': n_found,
        'return': ret, 'sharpe': sh, 'mdd': mdd,
        'test_equity': test_equity,
        'pool_pairs': [(p.y, p.x) for p in fixed_pool],
        'screen_time': screen_time,
    }
    pickle.dump(cache, open(CACHE, 'wb'))

    print(f'  W{w["idx"]:>2d} pool_full={n_found:>3d} used={n_used:>2d}  '
          f'Ret {ret:+7.2%}  Sh {sh:+5.2f}  MDD {mdd:+6.2%}  '
          f'[screen {screen_time:.0f}s, total {time.time()-t0:.0f}s]')

print()
print(f'완료. 캐시 키: {len(cache)}')
