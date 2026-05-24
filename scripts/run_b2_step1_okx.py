"""
B-2 STEP 1 (OKX 버전): 데이터 다운로드

Binance가 차단되어 OKX로 전환.

소스: OKX public API (인증 불필요)
  - https://www.okx.com/api/v5/market/candles (1D bars)
  - https://www.okx.com/api/v5/public/funding-rate-history (8h funding)

심볼:
  spot: BTC-USDT, ETH-USDT
  perp: BTC-USDT-SWAP, ETH-USDT-SWAP
"""
import sys, os, time, pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'

if os.path.exists(CACHE):
    cache = pickle.load(open(CACHE, 'rb'))
    print(f'기존 캐시 load: {list(cache.keys())}')
else:
    cache = {}


def okx_get(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise


def fetch_okx_candles(inst_id: str, bar: str = '1D',
                      start_ms: int = None, end_ms: int = None) -> pd.DataFrame:
    """
    OKX K-line. 응답이 최신부터 역순으로 옴, page는 'before'/'after' 사용.
    pagination: 한 번에 최대 100건. 과거로 갈 때 'after' 사용.
    """
    base = 'https://www.okx.com/api/v5/market/candles'
    history_base = 'https://www.okx.com/api/v5/market/history-candles'
    all_rows = []

    # 최근 데이터는 candles, 과거는 history-candles 써야 함
    # 단순화: history-candles만 사용 (역사 데이터)
    cur_after = str(end_ms) if end_ms else ''

    while True:
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': '100',
        }
        if cur_after:
            params['after'] = cur_after

        try:
            resp = okx_get(history_base, params)
        except Exception as e:
            print(f'  error: {e}')
            break

        data = resp.get('data', [])
        if not data:
            break

        # data는 최신→과거 순. ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm
        for row in data:
            ts = int(row[0])
            if start_ms and ts < start_ms:
                break
            all_rows.append({
                'ts': ts,
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5]),
            })

        oldest_ts = int(data[-1][0])
        if start_ms and oldest_ts <= start_ms:
            break
        if len(data) < 100:
            break

        cur_after = str(oldest_ts)
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('time').sort_index()
    df = df.drop(columns=['ts'])
    df = df[~df.index.duplicated(keep='last')]
    return df


def fetch_okx_funding(inst_id: str, start_ms: int = None,
                      end_ms: int = None) -> pd.DataFrame:
    """
    OKX funding rate history. 100건씩 페이징.
    """
    base = 'https://www.okx.com/api/v5/public/funding-rate-history'
    all_rows = []
    cur_after = str(end_ms) if end_ms else ''

    while True:
        params = {
            'instId': inst_id,
            'limit': '100',
        }
        if cur_after:
            params['after'] = cur_after

        try:
            resp = okx_get(base, params)
        except Exception as e:
            print(f'  funding error: {e}')
            break

        data = resp.get('data', [])
        if not data:
            break

        for row in data:
            ts = int(row['fundingTime'])
            if start_ms and ts < start_ms:
                break
            all_rows.append({
                'ts': ts,
                'fundingRate': float(row['fundingRate']),
                'realizedRate': float(row.get('realizedRate', row['fundingRate'])),
            })

        oldest_ts = int(data[-1]['fundingTime'])
        if start_ms and oldest_ts <= start_ms:
            break
        if len(data) < 100:
            break

        cur_after = str(oldest_ts)
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('time').sort_index()
    df = df.drop(columns=['ts'])
    df = df[~df.index.duplicated(keep='last')]
    return df


# ============================================================
SYMBOLS = {
    'BTC': {'spot': 'BTC-USDT', 'perp': 'BTC-USDT-SWAP'},
    'ETH': {'spot': 'ETH-USDT', 'perp': 'ETH-USDT-SWAP'},
}
START = datetime(2020, 1, 1)
END = datetime(2026, 5, 22)
START_MS = int(START.timestamp() * 1000)
END_MS = int(END.timestamp() * 1000)

print(f'다운로드 범위: {START.date()} ~ {END.date()}')

for name, sids in SYMBOLS.items():
    print()
    print(f'=== {name} ===')

    # Spot daily
    key = f'{name}_spot_1d'
    if key not in cache or len(cache.get(key, [])) < 100:
        print(f'  Spot 1d ({sids["spot"]})...')
        t0 = time.time()
        df = fetch_okx_candles(sids['spot'], '1D', START_MS, END_MS)
        if len(df) > 0:
            cache[key] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}행, {df.index[0].date()}~{df.index[-1].date()} [{time.time()-t0:.1f}s]')
        else:
            print(f'    → 데이터 없음 [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Spot 1d 캐시: {len(df)}행')

    # Perp daily
    key = f'{name}_perp_1d'
    if key not in cache or len(cache.get(key, [])) < 100:
        print(f'  Perp 1d ({sids["perp"]})...')
        t0 = time.time()
        df = fetch_okx_candles(sids['perp'], '1D', START_MS, END_MS)
        if len(df) > 0:
            cache[key] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}행, {df.index[0].date()}~{df.index[-1].date()} [{time.time()-t0:.1f}s]')
        else:
            print(f'    → 데이터 없음 [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Perp 1d 캐시: {len(df)}행')

    # Funding
    key = f'{name}_funding'
    if key not in cache or len(cache.get(key, [])) < 100:
        print(f'  Funding 8h...')
        t0 = time.time()
        df = fetch_okx_funding(sids['perp'], START_MS, END_MS)
        if len(df) > 0:
            cache[key] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}건, {df.index[0]}~{df.index[-1]} [{time.time()-t0:.1f}s]')
        else:
            print(f'    → 데이터 없음 [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Funding 캐시: {len(df)}건')


# ============================================================
# 데이터 요약
# ============================================================
print()
print('=' * 72)
print('데이터 요약')
print('=' * 72)

for name in SYMBOLS:
    spot_k = f'{name}_spot_1d'
    perp_k = f'{name}_perp_1d'
    fund_k = f'{name}_funding'
    if spot_k not in cache or perp_k not in cache or fund_k not in cache:
        print(f'\n{name}: 데이터 부족 — 다시 실행 필요')
        continue

    spot = cache[spot_k]
    perp = cache[perp_k]
    fund = cache[fund_k]

    print(f'\n{name}:')
    print(f'  Spot 1d  : {len(spot)}일, {spot.index[0].date()} ~ {spot.index[-1].date()}')
    print(f'  Perp 1d  : {len(perp)}일')
    print(f'  Funding  : {len(fund)}건 (8시간 단위)')

    fr = fund['fundingRate']
    print(f'    펀딩비 평균  : {fr.mean()*100:+.5f}% per 8h = {fr.mean()*100*3*365:+.2f}% 연환산')
    print(f'    중앙값       : {fr.median()*100:+.5f}%')
    print(f'    양수 비율    : {(fr > 0).mean()*100:.1f}%')
    print(f'    std          : {fr.std()*100:.5f}%')
    print(f'    [최소, 최대] : [{fr.min()*100:+.4f}%, {fr.max()*100:+.4f}%]')

    common_idx = spot.index.intersection(perp.index)
    if len(common_idx) > 0:
        basis = (perp.loc[common_idx, 'close'] - spot.loc[common_idx, 'close']) / spot.loc[common_idx, 'close']
        print(f'    Perp-Spot 베이시스 평균: {basis.mean()*100:+.4f}%, std {basis.std()*100:.4f}%')

print()
print(f'캐시 저장: {CACHE}')
