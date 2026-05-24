"""
B-2 STEP 2: 1h candle 데이터 보충 — rate limit 우회용 페이징

step1에서 candle이 4949건만 받아짐 (최근 7개월).
funding은 26K건 다 받았으니, 동일 기간 1h candle 필요.

전략:
  - startTime 변경하면서 5,000건씩 페이징
  - rate limit 시 sleep
"""
import os, time, pickle
import requests
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))


def hl_candles_paginated(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """1h candles, 5000건씩 페이징"""
    url = 'https://api.hyperliquid.xyz/info'
    all_rows = []
    cur = start_ms

    while cur < end_ms:
        payload = {
            'type': 'candleSnapshot',
            'req': {
                'coin': coin,
                'interval': '1h',
                'startTime': cur,
                'endTime': end_ms,
            },
        }
        success = False
        for attempt in range(5):
            try:
                r = requests.post(url, json=payload, timeout=20)
                if r.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f'    rate limit, wait {wait}s')
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                success = True
                break
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f'    err {e}, retry in {wait}s')
                time.sleep(wait)

        if not success or not data:
            print(f'    skip from {cur}')
            break

        all_rows.extend(data)
        latest_ts = max(d['t'] for d in data)
        if latest_ts <= cur or len(data) < 5000:
            if latest_ts > cur:
                cur = latest_ts + 1
            else:
                break
        else:
            cur = latest_ts + 1
        cur_ts = pd.Timestamp(cur, unit='ms')
        print(f'    progress: {len(all_rows)} rows, cursor {cur_ts}')
        time.sleep(2)  # rate limit 여유

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


import datetime
START_MS = int(datetime.datetime(2023, 5, 12).timestamp() * 1000)
END_MS = int(datetime.datetime(2026, 5, 22).timestamp() * 1000)

COINS = ['BTC', 'ETH']

for coin in COINS:
    key = f'HL_{coin}_1h_full'
    if key in cache and len(cache[key]) > 20000:
        print(f'{coin}: 캐시 {len(cache[key])}건 hit')
        continue
    print(f'\n{coin} 1h candle 다운로드...')
    t0 = time.time()
    df = hl_candles_paginated(coin, START_MS, END_MS)
    if len(df) > 0:
        cache[key] = df
        pickle.dump(cache, open(CACHE, 'wb'))
        print(f'  → {len(df)}건, {df.index[0]} ~ {df.index[-1]} [{time.time()-t0:.1f}s]')

print()
print('완료. 캐시:')
for k in cache:
    v = cache[k]
    if hasattr(v, '__len__'):
        print(f'  {k}: {len(v)}')
