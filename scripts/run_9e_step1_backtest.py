"""
9-E Step 1: 백테스트 실행 후 pickle 저장 (변형별)
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, Backtester, compute_spread,
)
from production_system import ProductionSystem


TICKERS = ['KO', 'PEP', 'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX', 'V', 'MA', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T', 'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW', 'WMT', 'TGT']

CACHE_PATH = '/tmp/mvp9e_cache.pkl'

# 기존 캐시 load
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print(f'캐시 load: {list(cache.keys())}')
else:
    cache = {}

# 데이터 (캐시)
if 'data' not in cache:
    print('데이터 다운로드...')
    data = yf.download(TICKERS, period='10y', interval='1d',
                       progress=False, auto_adjust=True)['Close']
    data = data.dropna(axis=1, thresh=int(len(data) * 0.95)).dropna()
    cache['data'] = data
else:
    data = cache['data']
print(f'데이터: {data.shape[0]}일 × {data.shape[1]}종목, '
      f'{data.index[0].date()} ~ {data.index[-1].date()}')

finder = PairsFinder(alpha_adf=0.10, pvalue_coint=0.05,
                     max_halflife=30.0, min_halflife=1.0)

# (b) 발굴 + 백테스트 (캐시)
if 'b' not in cache:
    print()
    print('(b) 발굴 + 백테스트...')
    t0 = time.time()
    train_split = int(len(data) * 0.7)
    fixed_pool = finder.screen_pairs(data.iloc[:train_split])
    print(f'  발굴 {len(fixed_pool)}개 페어 [{time.time()-t0:.1f}s]')
    for p in fixed_pool[:10]:
        print(f'    {p}')

    INITIAL_CAPITAL = 100_000
    daily_returns_b = pd.Series(0.0, index=data.index)
    equal_w = 0.10 / max(len(fixed_pool), 1)
    for p in fixed_pool:
        spread = compute_spread(data[p.y], data[p.x], p.beta)
        sw = max(20, min(60, int(p.half_life * 2)))
        sg = SignalGenerator(window=sw, entry=2.0, exit_thr=0.5, stop=3.5)
        z, pos, fc = sg.generate(spread)
        bt = Backtester(initial_capital=INITIAL_CAPITAL,
                        capital_fraction=1.0,
                        fee_rate=0.0004, slippage=0.0005)
        res = bt.run(data[p.y], data[p.x], p.beta, pos, fc)
        daily_returns_b += equal_w * res.equity.pct_change().fillna(0)

    equity_b = INITIAL_CAPITAL * (1 + daily_returns_b).cumprod()
    cache['b'] = {
        'equity': equity_b,
        'fixed_pool_size': len(fixed_pool),
        'fixed_pool_pairs': [(p.y, p.x) for p in fixed_pool],
    }
    print(f'  (b) 최종 ${equity_b.iloc[-1]:,.0f} [{time.time()-t0:.1f}s total]')

# 변형 정의
COMMON = dict(
    finder=finder,
    lookback_days=365,
    sig_entry=2.0, sig_exit=0.5, sig_stop=3.5,
    initial_capital=100_000,
    capital_per_pair_cap=0.10,
    portfolio_kelly_fraction=0.25,
    use_history_for_kelly=True,
    max_active_pairs=5,
    min_historical_sharpe=0.0,
    quality_lookback=90,
)

VARIANTS = [
    ('r180_cd0',  {'refresh_every_days': 180, 'pair_cooldown_days': 0}),
    ('r180_cd30', {'refresh_every_days': 180, 'pair_cooldown_days': 30}),
    ('r90_cd30',  {'refresh_every_days': 90,  'pair_cooldown_days': 30}),
    ('r30_cd30',  {'refresh_every_days': 30,  'pair_cooldown_days': 30}),
    ('r30_cd0',   {'refresh_every_days': 30,  'pair_cooldown_days': 0}),
]

# 변형별 백테스트 (캐시된 건 skip)
INITIAL_CAPITAL = 100_000
for label, params in VARIANTS:
    if label in cache:
        print(f'  {label}: 캐시 hit, skip')
        continue
    print()
    print(f'{label}: 백테스트 시작 (refresh={params["refresh_every_days"]}, cd={params["pair_cooldown_days"]})...')
    t0 = time.time()
    sys_v = ProductionSystem(**COMMON, **params)
    res = sys_v.run(data, verbose=False)
    eq = res.equity_curve
    n_snapshots = len(res.monthly_states)
    avg_pairs = np.mean([len(s.active_pairs) for s in res.monthly_states
                         if s.active_pairs]) if any(s.active_pairs for s in res.monthly_states) else 0
    prev_pids = set()
    n_new = 0
    for st in res.monthly_states:
        cur_pids = {f'{p.y}~{p.x}' for p in st.active_pairs}
        n_new += len(cur_pids - prev_pids)
        prev_pids = cur_pids

    ret = eq.iloc[-1] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())

    cache[label] = {
        'equity': eq, 'return': ret, 'sharpe': sharpe, 'mdd': mdd,
        'n_snapshots': n_snapshots, 'avg_pairs': avg_pairs, 'n_new': n_new,
        'params': params,
    }
    # 즉시 캐시 저장 (중단되어도 진척 보존)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f'  → ${eq.iloc[-1]:,.0f}  Ret {ret:+.2%}  Sh {sharpe:+.2f}  '
          f'MDD {mdd:+.2%}  snap {n_snapshots} pairs {avg_pairs:.1f} '
          f'[{time.time()-t0:.1f}s]')

print()
print(f'완료. 캐시 키: {list(cache.keys())}')
