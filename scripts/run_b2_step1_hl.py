"""
B-2 STEP 1c: Hyperliquid 펀딩비 다운로드

OKX는 90일까지만 제공. Binance는 차단. 
Hyperliquid가 무료 + 인증 없이 2023-05부터 풍부한 데이터 제공.

데이터:
  - 펀딩비 (1시간 단위, Hyperliquid 특성)
  - 가격 (1시간 후보, 일봉 가공)
"""
import sys, os, time, pickle
import requests
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'
cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}


def hyperliquid_funding(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Hyperliquid 펀딩비 다운로드. 500건씩, 1시간 간격.
    """
    url = 'https://api.hyperliquid.xyz/info'
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        payload = {
            'type': 'fundingHistory',
            'coin': coin,
            'startTime': cur_start,
            'endTime': end_ms,
        }
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  error: {e}, retry')
            time.sleep(2)
            continue

        if not data:
            break

        all_rows.extend(data)
        latest_ts = max(d['time'] for d in data)
        if latest_ts <= cur_start:
            break  # no progress
        cur_start = latest_ts + 1
        if len(data) < 500:
            break
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['premium'] = df['premium'].astype(float)
    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['fundingRate', 'premium']]


def hyperliquid_candles(coin: str, interval: str, start_ms: int,
                        end_ms: int) -> pd.DataFrame:
    """
    Hyperliquid candle snapshot. interval: '1d', '1h', etc.
    한 번에 5000건 max.
    """
    url = 'https://api.hyperliquid.xyz/info'
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        payload = {
            'type': 'candleSnapshot',
            'req': {
                'coin': coin,
                'interval': interval,
                'startTime': cur_start,
                'endTime': end_ms,
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  candle error: {e}')
            time.sleep(2)
            continue

        if not data:
            break

        all_rows.extend(data)
        latest_ts = max(d['t'] for d in data)
        if latest_ts <= cur_start:
            break
        cur_start = latest_ts + 1
        if len(data) < 5000:
            break
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['time'] = pd.to_datetime(df['t'], unit='ms')
    df['open'] = df['o'].astype(float)
    df['high'] = df['h'].astype(float)
    df['low'] = df['l'].astype(float)
    df['close'] = df['c'].astype(float)
    df['volume'] = df['v'].astype(float)
    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['open', 'high', 'low', 'close', 'volume']]


# ============================================================
import datetime
START_MS = int(datetime.datetime(2023, 5, 1).timestamp() * 1000)
END_MS = int(datetime.datetime(2026, 5, 22).timestamp() * 1000)

COINS = ['BTC', 'ETH', 'SOL']  # SOL 추가 — 변동성 더 큼, 펀딩비 더 큼

for coin in COINS:
    print()
    print(f'=== {coin} (Hyperliquid) ===')

    key_funding = f'HL_{coin}_funding'
    if key_funding not in cache:
        print(f'  Funding...')
        t0 = time.time()
        df = hyperliquid_funding(coin, START_MS, END_MS)
        if len(df) > 0:
            cache[key_funding] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}건, {df.index[0]} ~ {df.index[-1]} [{time.time()-t0:.1f}s]')
        else:
            print(f'    → 데이터 없음')
    else:
        df = cache[key_funding]
        print(f'  Funding 캐시: {len(df)}건')

    key_candle = f'HL_{coin}_1h'
    if key_candle not in cache:
        print(f'  1h candle...')
        t0 = time.time()
        df = hyperliquid_candles(coin, '1h', START_MS, END_MS)
        if len(df) > 0:
            cache[key_candle] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}건, {df.index[0]} ~ {df.index[-1]} [{time.time()-t0:.1f}s]')
        else:
            print(f'    → 데이터 없음')
    else:
        df = cache[key_candle]
        print(f'  1h candle 캐시: {len(df)}건')


# ============================================================
print()
print('=' * 72)
print('Hyperliquid 데이터 요약')
print('=' * 72)

for coin in COINS:
    fk = f'HL_{coin}_funding'
    ck = f'HL_{coin}_1h'
    if fk not in cache or ck not in cache:
        continue
    fund = cache[fk]
    cdl = cache[ck]

    print(f'\n{coin}:')
    print(f'  Funding: {len(fund)}건 ({fund.index[0].date()} ~ {fund.index[-1].date()})')
    print(f'  Candle : {len(cdl)}건 1h')

    # Hyperliquid: 1h funding
    fr = fund['fundingRate']
    annual_factor = 24 * 365  # 1h마다 funding
    print(f'  funding 평균  : {fr.mean()*100:+.6f}% per 1h = {fr.mean()*100*annual_factor:+.2f}% 연환산')
    print(f'  중앙값         : {fr.median()*100:+.6f}%')
    print(f'  양수 비율      : {(fr > 0).mean()*100:.1f}%')
    print(f'  std            : {fr.std()*100:.6f}%')
    print(f'  [최소, 최대]   : [{fr.min()*100:+.5f}%, {fr.max()*100:+.5f}%]')
    print(f'  음수 비율      : {(fr < 0).mean()*100:.1f}%')

print()
print(f'캐시 저장: {CACHE}')
