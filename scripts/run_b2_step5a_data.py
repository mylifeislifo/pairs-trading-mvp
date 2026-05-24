"""
B-2 STEP 5a: OKX 1h 데이터 다운로드 (7개월)

목적: HL 1h funding과 동일 기간 OKX 1h spot+perp 가격 → 정밀 시뮬 가능
"""
import os, time, pickle, requests
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

CACHE = '/tmp/mvp_b2_cache.pkl'
cache = pickle.load(open(CACHE, 'rb'))


def fetch_okx_1h(inst_id: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX 1h history-candles, 100건씩 페이징."""
    url = 'https://www.okx.com/api/v5/market/history-candles'
    all_rows = []
    cur_after = str(end_ms)
    iter_count = 0
    max_iter = 100  # 안전장치

    while iter_count < max_iter:
        iter_count += 1
        params = {'instId': inst_id, 'bar': '1H', 'limit': '100', 'after': cur_after}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            d = r.json().get('data', [])
        except Exception as e:
            print(f'  err {e}, retry in 2s')
            time.sleep(2)
            continue
        if not d:
            break

        oldest_this_batch = int(d[-1][0])
        for row in d:
            ts = int(row[0])
            if ts < start_ms:
                continue
            all_rows.append({
                'ts': ts,
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
            })

        if oldest_this_batch <= start_ms or len(d) < 100:
            break
        cur_after = str(oldest_this_batch)
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['open', 'high', 'low', 'close']]


# Hyperliquid 1h candle이 있는 기간과 정확히 일치 시킴
hl_btc_1h = cache['HL_BTC_1h_full']
start = hl_btc_1h.index[0]
end = hl_btc_1h.index[-1]
print(f'대상 기간: {start} ~ {end}')

start_ms = int(start.timestamp() * 1000) - 3600_000  # 1h 여유
end_ms = int(end.timestamp() * 1000) + 3600_000

for coin, (spot_id, perp_id) in [
    ('BTC', ('BTC-USDT', 'BTC-USDT-SWAP')),
    ('ETH', ('ETH-USDT', 'ETH-USDT-SWAP')),
]:
    print(f'\n{coin}:')
    for kind, inst in [('spot', spot_id), ('perp', perp_id)]:
        key = f'OKX_{coin}_{kind}_1h'
        if key in cache and len(cache[key]) > 4000:
            print(f'  {kind}: 캐시 {len(cache[key])}건')
            continue
        print(f'  {kind} 1h ({inst}) 다운로드...')
        t0 = time.time()
        df = fetch_okx_1h(inst, start_ms, end_ms)
        if len(df) > 0:
            cache[key] = df
            pickle.dump(cache, open(CACHE, 'wb'))
            print(f'    → {len(df)}건, {df.index[0]} ~ {df.index[-1]} [{time.time()-t0:.1f}s]')

print()
print('완료. 캐시:')
for k in sorted(cache.keys()):
    if 'OKX' in k and '1h' in k:
        print(f'  {k}: {len(cache[k])}건')
