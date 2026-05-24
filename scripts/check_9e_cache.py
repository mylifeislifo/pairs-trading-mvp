import pickle
c = pickle.load(open('/tmp/mvp9e_cache.pkl', 'rb'))
print('Cache keys:', list(c.keys()))
for k in c:
    if k == 'data':
        print(f'  {k}: shape={c[k].shape}')
    elif k == 'b':
        eq = c[k]['equity']
        print(f'  {k}: pool_size={c[k]["fixed_pool_size"]}, final=${eq.iloc[-1]:,.0f}')
    else:
        r = c[k]
        print(f'  {k}: Ret={r["return"]:+.2%}, Sh={r["sharpe"]:+.2f}, MDD={r["mdd"]:+.2%}, snap={r["n_snapshots"]}')
