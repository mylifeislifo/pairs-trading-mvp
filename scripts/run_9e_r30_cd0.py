"""9-E: r30_cd0 단독 실행"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder
from production_system import ProductionSystem

CACHE = '/tmp/mvp9e_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))
data = cache['data']

if 'r30_cd0' in cache:
    print('이미 캐시 있음, skip')
    sys.exit(0)

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)
COMMON = dict(finder=finder, lookback_days=365, sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
              initial_capital=100000, capital_per_pair_cap=0.10,
              portfolio_kelly_fraction=0.25, use_history_for_kelly=True,
              max_active_pairs=5, min_historical_sharpe=0.0, quality_lookback=90)

label = 'r30_cd0'
params = {'refresh_every_days': 30, 'pair_cooldown_days': 0}
print(f'{label} 시작...')
t0 = time.time()
sys_v = ProductionSystem(**COMMON, **params)
res = sys_v.run(data, verbose=False)
eq = res.equity_curve
ret = eq.iloc[-1]/eq.iloc[0] - 1
daily = eq.pct_change().dropna()
sh = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
mdd = float((eq/eq.cummax()-1).min())
n_snap = len(res.monthly_states)
avg = np.mean([len(s.active_pairs) for s in res.monthly_states if s.active_pairs]) if any(s.active_pairs for s in res.monthly_states) else 0
prev = set(); n_new = 0
for st in res.monthly_states:
    cur = {f'{x.y}~{x.x}' for x in st.active_pairs}
    n_new += len(cur - prev); prev = cur
cache[label] = {'equity': eq, 'return': ret, 'sharpe': sh, 'mdd': mdd,
                'n_snapshots': n_snap, 'avg_pairs': avg, 'n_new': n_new, 'params': params}
pickle.dump(cache, open(CACHE, 'wb'))
print(f'{label}: final=${eq.iloc[-1]:,.0f} Ret={ret:+.2%} Sh={sh:+.2f} MDD={mdd:+.2%} snap={n_snap} pairs={avg:.1f} [{time.time()-t0:.1f}s]')
