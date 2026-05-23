"""
페어 트레이딩 MVP 실전 실행 스크립트
=====================================

전체 흐름:
  1. yfinance로 미국 대형주 가격 다운로드 (3년)
  2. Train/Test 분할 (look-ahead 차단)
  3. Train에서 페어 발굴 (§1 + §2)
  4. Train Z-score 신호로 켈리 비율 추정 (§4)
  5. Test 구간에서 OOS 백테스트
  6. 결과 시각화 + 리포트
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, KellySizer, Backtester,
    compute_spread
)


# ============================================================
# 1. 데이터 다운로드
# ============================================================
# 후보군: 클래식한 페어 트레이딩 종목들
#   - 소비재: KO, PEP
#   - 금/은 ETF: GLD, SLV, GDX, GDXJ
#   - 메이저: AAPL, MSFT, GOOG, META, AMZN
#   - 반도체: NVDA, AMD, INTC, TSM
#   - 항공: DAL, UAL, AAL, LUV
#   - 결제: V, MA, PYPL
#   - 통신: VZ, T, TMUS
TICKERS = [
    'KO', 'PEP',
    'GLD', 'SLV', 'GDX', 'GDXJ',
    'XOM', 'CVX',
    'V', 'MA',
    'AAPL', 'MSFT',
    'NVDA', 'AMD',
    'DAL', 'UAL', 'AAL', 'LUV',
    'VZ', 'T',
    'JPM', 'BAC', 'C', 'WFC',
    'HD', 'LOW',
    'WMT', 'TGT',
]

PERIOD = '3y'
INTERVAL = '1d'

print('=' * 70)
print('1. 데이터 다운로드')
print('=' * 70)
print(f'  티커: {len(TICKERS)}종, 기간: {PERIOD}, 봉: {INTERVAL}')

data = yf.download(TICKERS, period=PERIOD, interval=INTERVAL,
                   progress=False, auto_adjust=True)['Close']
data = data.dropna(axis=1, thresh=int(len(data) * 0.95))
data = data.dropna()
print(f'  최종 데이터: {data.shape[0]}일 x {data.shape[1]}종목')
print(f'  기간: {data.index[0].date()} ~ {data.index[-1].date()}')


# ============================================================
# 2. Train/Test 분할
# ============================================================
TRAIN_RATIO = 0.7
cut = int(len(data) * TRAIN_RATIO)
train_data = data.iloc[:cut]
test_data = data.iloc[cut:]

print()
print('=' * 70)
print('2. Train/Test 분할 (시간 순)')
print('=' * 70)
print(f'  Train: {train_data.index[0].date()} ~ {train_data.index[-1].date()} '
      f'({len(train_data)}일)')
print(f'  Test:  {test_data.index[0].date()} ~ {test_data.index[-1].date()} '
      f'({len(test_data)}일)')


# ============================================================
# 3. 페어 발굴 (Train 구간만 사용)
# ============================================================
print()
print('=' * 70)
print('3. 페어 발굴 — §1 ADF + §2 공적분 + Half-life + TLS')
print('=' * 70)

finder = PairsFinder(
    alpha_adf=0.10,        # 약간 완화 (실제 시장 ADF는 보수적)
    pvalue_coint=0.05,
    max_halflife=30.0,
    min_halflife=1.0,
)

pairs = finder.screen_pairs(train_data)
print(f'  적격 페어: {len(pairs)}개')
print()
print('  상위 10개 페어:')
for i, p in enumerate(pairs[:10], 1):
    print(f'  {i:2d}. {p}')


# ============================================================
# 4. 최적 페어 선택 및 OOS 백테스트
# ============================================================
if not pairs:
    print('\n[중단] 적격 페어 없음. 임계치 조정 또는 티커 추가 필요.')
    raise SystemExit(0)

# Sharpe 기준으로 상위 페어들을 in-sample에서 평가하고 OOS로 검증
print()
print('=' * 70)
print('4. In-sample 평가 (켈리 비율 추정)')
print('=' * 70)

candidate_results = []
for p in pairs[:10]:
    try:
        # Train 구간 spread
        train_spread = compute_spread(
            train_data[p.y], train_data[p.x], p.beta)

        # 윈도우는 half-life의 약 2배
        window = max(20, min(60, int(p.half_life * 2)))

        sig_gen = SignalGenerator(window=window, entry=2.0,
                                   exit_thr=0.5, stop=3.5)
        z_tr, pos_tr, fc_tr = sig_gen.generate(train_spread)

        # Train 백테스트 (capital_fraction=1.0으로 raw 수익률 추출)
        bt_tr = Backtester(initial_capital=100_000,
                           capital_fraction=1.0,
                           fee_rate=0.0004,
                           slippage=0.0005)
        res_tr = bt_tr.run(train_data[p.y], train_data[p.x],
                           p.beta, pos_tr, fc_tr)

        if res_tr.trades.empty or len(res_tr.trades) < 5:
            continue

        # 켈리 비율: in-sample 거래 통계에서
        kelly_f = KellySizer.from_trades(
            res_tr.trades['pnl_pct'],
            fraction=0.25, cap=0.20)

        candidate_results.append({
            'pair': p,
            'window': window,
            'train_sharpe': res_tr.metrics.get('sharpe', 0),
            'train_n_trades': res_tr.metrics.get('n_trades', 0),
            'train_win_rate': res_tr.metrics.get('win_rate', 0),
            'kelly_f': kelly_f,
        })
    except Exception as e:
        print(f'  Skip {p.y}~{p.x}: {e}')

candidate_results.sort(key=lambda r: r['train_sharpe'], reverse=True)

print()
print('  In-sample 상위 5개:')
print(f'  {"Pair":<20s} {"Sharpe":>8s} {"Trades":>8s} {"WinRate":>8s} {"Kelly":>8s}')
for r in candidate_results[:5]:
    print(f'  {r["pair"].y}~{r["pair"].x:<15s} '
          f'{r["train_sharpe"]:8.2f} '
          f'{r["train_n_trades"]:8d} '
          f'{r["train_win_rate"]:8.2%} '
          f'{r["kelly_f"]:8.2%}')


# ============================================================
# 5. OOS 백테스트 (Test 구간)
# ============================================================
print()
print('=' * 70)
print('5. OOS 백테스트 (Test 구간)')
print('=' * 70)

# 켈리 양수 + Sharpe > 1 후보만 OOS 진입
qualified = [r for r in candidate_results
             if r['kelly_f'] > 0 and r['train_sharpe'] > 1.0]

if not qualified:
    print('  [경고] In-sample Sharpe>1 + Kelly>0 페어 없음. 상위 3개로 진행.')
    qualified = candidate_results[:3]

oos_results = []
for r in qualified[:5]:
    p = r['pair']
    try:
        test_spread = compute_spread(
            test_data[p.y], test_data[p.x], p.beta)

        sig_gen = SignalGenerator(window=r['window'], entry=2.0,
                                   exit_thr=0.5, stop=3.5)
        z_te, pos_te, fc_te = sig_gen.generate(test_spread)

        bt_te = Backtester(initial_capital=100_000,
                           capital_fraction=r['kelly_f'],
                           fee_rate=0.0004,
                           slippage=0.0005)
        res_te = bt_te.run(test_data[p.y], test_data[p.x],
                           p.beta, pos_te, fc_te)

        oos_results.append({
            'pair': p,
            'kelly_f': r['kelly_f'],
            'result': res_te,
        })
    except Exception as e:
        print(f'  Skip {p.y}~{p.x}: {e}')

print()
print(f'  {"Pair":<20s} {"Return":>10s} {"Sharpe":>8s} {"MDD":>8s} '
      f'{"Trades":>8s} {"WinRate":>8s}')
for r in oos_results:
    m = r['result'].metrics
    print(f'  {r["pair"].y}~{r["pair"].x:<15s} '
          f'{m.get("total_return", 0):10.2%} '
          f'{m.get("sharpe", 0):8.2f} '
          f'{m.get("max_drawdown", 0):8.2%} '
          f'{m.get("n_trades", 0):8d} '
          f'{m.get("win_rate", 0):8.2%}')


# ============================================================
# 6. 시각화 (상위 페어 1개)
# ============================================================
if oos_results:
    print()
    print('=' * 70)
    print('6. 시각화')
    print('=' * 70)

    # Sharpe 가장 좋은 OOS 페어
    oos_results.sort(key=lambda x: x['result'].metrics.get('sharpe', 0),
                     reverse=True)
    best = oos_results[0]
    p = best['pair']
    res = best['result']

    # Train + Test 통합 시각화
    full_spread = compute_spread(data[p.y], data[p.x], p.beta)
    window = max(20, min(60, int(p.half_life * 2)))
    sig_gen = SignalGenerator(window=window, entry=2.0,
                               exit_thr=0.5, stop=3.5)
    z_full, pos_full, fc_full = sig_gen.generate(full_spread)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # (a) 가격
    ax = axes[0]
    ax.plot(data.index, data[p.y], label=p.y, color='steelblue', lw=1.5)
    ax2 = ax.twinx()
    ax2.plot(data.index, data[p.x], label=p.x, color='darkorange', lw=1.5)
    ax.axvline(test_data.index[0], color='red', ls='--', alpha=0.5,
               label='Train/Test split')
    ax.set_ylabel(f'{p.y} (USD)', color='steelblue')
    ax2.set_ylabel(f'{p.x} (USD)', color='darkorange')
    ax.set_title(f'Pair: {p.y} ~ {p.x}  (beta={p.beta:.3f}, HL={p.half_life:.1f}d, '
                 f'p={p.pvalue:.4f})')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (b) Spread
    ax = axes[1]
    ax.plot(full_spread.index, full_spread, color='black', lw=1)
    ax.axhline(full_spread.iloc[:cut].mean(), color='gray', ls='--', alpha=0.5,
               label='Train mean')
    ax.axvline(test_data.index[0], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('log Spread')
    ax.set_title('Spread = log(y) - beta * log(x)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (c) Z-score
    ax = axes[2]
    ax.plot(z_full.index, z_full, color='purple', lw=1)
    ax.axhline(2.0, color='red', ls=':', alpha=0.7, label='Entry +-2')
    ax.axhline(-2.0, color='green', ls=':', alpha=0.7)
    ax.axhline(3.5, color='red', ls='--', alpha=0.5, label='Stop +-3.5')
    ax.axhline(-3.5, color='red', ls='--', alpha=0.5)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(test_data.index[0], color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Z-score')
    ax.set_title(f'Z-score (window={window}, entry=+-2, stop=+-3.5)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (d) OOS Equity Curve
    ax = axes[3]
    ax.plot(res.equity.index, res.equity, color='darkgreen', lw=1.5,
            label='Pair Strategy')
    # Buy & Hold 비교 (50:50)
    bh = 100_000 * (data[[p.y, p.x]].loc[res.equity.index] /
                    data[[p.y, p.x]].loc[res.equity.index].iloc[0]).mean(axis=1)
    ax.plot(bh.index, bh, color='gray', lw=1, ls='--',
            label='Buy & Hold 50/50')
    ax.set_ylabel('Equity (USD)')
    ax.set_xlabel('Date')
    ax.set_title(f'OOS Equity Curve (Kelly fraction={best["kelly_f"]:.2%})')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    out_path = '/home/claude/pairs_backtest_chart.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  차트 저장: {out_path}')

    # 최종 리포트
    print()
    print('=' * 70)
    print(f'7. 최종 리포트: {p.y} ~ {p.x}')
    print('=' * 70)
    m = res.metrics
    print(f'  In-sample Sharpe   : {[r["train_sharpe"] for r in candidate_results if r["pair"] == p][0]:.2f}')
    print(f'  In-sample Kelly f  : {best["kelly_f"]:.2%}')
    print(f'  ---')
    print(f'  OOS 총수익률        : {m["total_return"]:8.2%}')
    print(f'  OOS CAGR           : {m["cagr"]:8.2%}')
    print(f'  OOS Sharpe         : {m["sharpe"]:8.2f}')
    print(f'  OOS Max Drawdown   : {m["max_drawdown"]:8.2%}')
    print(f'  OOS Calmar         : {m["calmar"]:8.2f}')
    print(f'  OOS 거래 횟수       : {m["n_trades"]}')
    print(f'  OOS 승률            : {m["win_rate"]:8.2%}')
    print(f'  OOS 평균 보유일      : {m["avg_trade_duration"]:.1f}일')
    print(f'  최종 자본          : ${res.equity.iloc[-1]:,.2f} (초기 $100,000)')
