"""
B-2 크립토 펀딩비 차익 — STEP 1: 데이터 다운로드

필요 데이터 (Binance):
  1. 현물 일봉 (BTCUSDT, ETHUSDT)
  2. 무기한 선물 일봉
  3. 펀딩비 기록 (8시간 단위)

소스: Binance public API (인증 불필요)
  - https://data.binance.vision/ (역사 데이터)
  - https://fapi.binance.com (펀딩비 직접)

저장: /tmp/mvp_b2_cache.pkl
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


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                 endpoint: str = 'spot') -> pd.DataFrame:
    """
    Binance K-line (일봉) 다운로드.
    endpoint: 'spot' (https://api.binance.com) or 'futures' (https://fapi.binance.com)
    """
    if endpoint == 'spot':
        base = 'https://api.binance.com/api/v3/klines'
    else:
        base = 'https://fapi.binance.com/fapi/v1/klines'

    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': cur,
            'endTime': end_ms,
            'limit': 1000,
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  fetch error: {e}, retry in 2s')
            time.sleep(2)
            continue
        if not data:
            break
        rows.extend(data)
        last_close_time = data[-1][6]
        cur = last_close_time + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)  # rate limit 여유

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'taker_base', 'taker_quote', 'ignore',
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df = df.set_index('open_time').sort_index()
    df.index = df.index.tz_localize(None)  # tz-naive
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Binance funding rate history.
    8시간마다 1 rate. 데이터 시작: 2019-09-08 (BTCUSDT)
    """
    base = 'https://fapi.binance.com/fapi/v1/fundingRate'
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            'symbol': symbol,
            'startTime': cur,
            'endTime': end_ms,
            'limit': 1000,
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  funding fetch error: {e}, retry in 2s')
            time.sleep(2)
            continue
        if not data:
            break
        rows.extend(data)
        last_time = data[-1]['fundingTime']
        cur = last_time + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df = df.set_index('fundingTime').sort_index()
    df.index = df.index.tz_localize(None)
    return df[['fundingRate']]


# ============================================================
# 다운로드 계획
# ============================================================
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
# 무기한 선물은 2019-09부터 풍부 (BTC), ETH는 2019-11부터
START = datetime(2020, 1, 1)
END = datetime(2026, 5, 22)
START_MS = int(START.timestamp() * 1000)
END_MS = int(END.timestamp() * 1000)

print(f'다운로드 범위: {START.date()} ~ {END.date()}')
print()

for sym in SYMBOLS:
    print(f'=== {sym} ===')

    # Spot daily
    key = f'{sym}_spot_1d'
    if key not in cache:
        print(f'  Spot 1d 다운로드...')
        t0 = time.time()
        df = fetch_klines(sym, '1d', START_MS, END_MS, 'spot')
        cache[key] = df
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'    → {len(df)}행, {df.index[0].date()}~{df.index[-1].date()} [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Spot 1d 캐시: {len(df)}행')

    # Futures daily
    key = f'{sym}_perp_1d'
    if key not in cache:
        print(f'  Perp 1d 다운로드...')
        t0 = time.time()
        df = fetch_klines(sym, '1d', START_MS, END_MS, 'futures')
        cache[key] = df
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'    → {len(df)}행, {df.index[0].date()}~{df.index[-1].date()} [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Perp 1d 캐시: {len(df)}행')

    # Funding rates (8시간 단위)
    key = f'{sym}_funding'
    if key not in cache:
        print(f'  Funding rates 다운로드...')
        t0 = time.time()
        df = fetch_funding(sym, START_MS, END_MS)
        cache[key] = df
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'    → {len(df)}행, {df.index[0]}~{df.index[-1]} [{time.time()-t0:.1f}s]')
    else:
        df = cache[key]
        print(f'  Funding 캐시: {len(df)}행')
    print()


# ============================================================
# 데이터 요약
# ============================================================
print('=' * 72)
print('데이터 요약')
print('=' * 72)

for sym in SYMBOLS:
    spot = cache[f'{sym}_spot_1d']
    perp = cache[f'{sym}_perp_1d']
    fund = cache[f'{sym}_funding']

    print(f'\n{sym}:')
    print(f'  Spot 1d  : {len(spot)}일, {spot.index[0].date()} ~ {spot.index[-1].date()}')
    print(f'  Perp 1d  : {len(perp)}일')
    print(f'  Funding  : {len(fund)}건 (8시간 단위)')

    # 펀딩비 통계
    fr = fund['fundingRate']
    print(f'    펀딩비 평균  : {fr.mean()*100:+.5f}% per 8h = {fr.mean()*100*3*365:+.2f}% 연환산')
    print(f'    펀딩비 중앙값: {fr.median()*100:+.5f}%')
    print(f'    양수 비율    : {(fr > 0).mean()*100:.1f}%')
    print(f'    표준편차     : {fr.std()*100:.5f}%')
    print(f'    최대         : {fr.max()*100:+.4f}%')
    print(f'    최소         : {fr.min()*100:+.4f}%')

    # 베이시스 (spot vs perp close)
    common_idx = spot.index.intersection(perp.index)
    if len(common_idx) > 0:
        basis = (perp.loc[common_idx, 'close'] - spot.loc[common_idx, 'close']) / spot.loc[common_idx, 'close']
        print(f'    Perp - Spot 베이시스 평균: {basis.mean()*100:+.4f}%, std {basis.std()*100:.4f}%')

print()
print(f'캐시 저장: {CACHE}')
