"""
10차 MVP-A: 확장 universe로 (b) walk-forward 재검증

9-C에서 확립된 결론:
  28 종목 + 단순 z-score → 어떤 변형도 통계적 alpha 부재

가설:
  Universe 부족이 진짜 문제. 더 큰 종목 풀에서는 (b) 고정 페어 풀이
  walk-forward에서도 양수 alpha를 보일 수 있다.

STEP 1: 80종목 데이터 다운로드 + universe 캐시
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp10a_cache.pkl'

# Universe: S&P 100 상위 + 주요 ETF
LARGE_CAPS = [
    # Tech & comms
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    'CRM', 'CSCO', 'ADBE', 'AMD', 'QCOM', 'TXN', 'INTC', 'IBM', 'INTU',
    'NFLX', 'CMCSA', 'TMUS', 'VZ', 'T',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP', 'BLK', 'SCHW',
    'C', 'USB', 'CB', 'PNC',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE',
    'AMGN', 'GILD', 'BMY', 'CVS', 'MDT',
    # Consumer
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'LOW',
    'SBUX', 'TGT', 'DIS',
    # Industrial & energy
    'XOM', 'CVX', 'COP', 'BA', 'HON', 'GE', 'UPS', 'RTX', 'CAT',
    'DE', 'LMT', 'NOC',
    # Others
    'NEE', 'LIN', 'SO', 'MMM',
]
ETFS = [
    # Broad market
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    # Sector
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
    # Commodities
    'GLD', 'SLV', 'GDX', 'GDXJ', 'USO',
]

TICKERS = LARGE_CAPS + ETFS
print(f'Target universe: {len(LARGE_CAPS)} large caps + {len(ETFS)} ETFs = {len(TICKERS)} tickers')

if os.path.exists(CACHE):
    cache = pickle.load(open(CACHE, 'rb'))
    if 'data' in cache:
        data = cache['data']
        print(f'캐시 hit: {data.shape[0]}일 × {data.shape[1]}종목')
        print(f'기간: {data.index[0].date()} ~ {data.index[-1].date()}')
        print(f'살아남은 종목: {list(data.columns)}')
        sys.exit(0)
else:
    cache = {}

print()
print('데이터 다운로드 중...')
t0 = time.time()
data = yf.download(TICKERS, period='10y', interval='1d',
                   progress=False, auto_adjust=True)['Close']
print(f'다운로드 완료 [{time.time()-t0:.1f}s]: 원본 {data.shape}')

# 결측치 처리 — 95% 이상 데이터 있는 종목만
data = data.dropna(axis=1, thresh=int(len(data) * 0.95))
print(f'95%+ 데이터 있는 종목: {data.shape[1]}개')

# 결측 행 제거
data = data.dropna()
print(f'완전 데이터: {data.shape[0]}일 × {data.shape[1]}종목')
print(f'기간: {data.index[0].date()} ~ {data.index[-1].date()}')

dropped = [t for t in TICKERS if t not in data.columns]
if dropped:
    print(f'제외된 종목 ({len(dropped)}개): {dropped}')

# Universe 분류 (분석용)
final_large_caps = [t for t in LARGE_CAPS if t in data.columns]
final_etfs = [t for t in ETFS if t in data.columns]
print(f'최종: {len(final_large_caps)} large caps + {len(final_etfs)} ETFs')

cache['data'] = data
cache['large_caps'] = final_large_caps
cache['etfs'] = final_etfs
cache['tickers'] = list(data.columns)

# 페어 후보 수
n = len(data.columns)
n_pairs = n * (n - 1) // 2
print(f'페어 후보: {n_pairs}개 (9-C의 378개 대비 {n_pairs/378:.1f}배)')

pickle.dump(cache, open(CACHE, 'wb'))
print()
print(f'캐시 저장: {CACHE}')
