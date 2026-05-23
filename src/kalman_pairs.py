"""
Kalman Dynamic Beta — 명세서 v2 §7
====================================

핵심 가설:
  1차 MVP의 GDX~KO 실패는 spread 평균값 시프트(-5.38→-4.8)가 원인이었다.
  정적 β=2.177이 시간 변화를 못 따라간 게 본질.
  Kalman β_t가 시변하여 시프트를 실시간 추적하면 어떻게 되는가?

상태 방정식: x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
관측 방정식: z_t = H x_t + v_t,        v_t ~ N(0, R)

페어 적용 (Pair 헤지비율 추정):
  - 잠재 상태: β_t (시변 헤지비율)
  - 상태 동역학: F = 1 (랜덤워크)
  - 관측: log(y_t) = β_t * log(x_t) + ε_t
  - H_t = log(x_t)

재귀 갱신:
  예측: β̂_{t|t-1} = β̂_{t-1|t-1}
        P_{t|t-1} = P_{t-1|t-1} + Q
  갱신: K_t = P_{t|t-1} H_t / (H_t² P_{t|t-1} + R)
        β̂_{t|t} = β̂_{t|t-1} + K_t (z_t - H_t β̂_{t|t-1})
        P_{t|t} = (1 - K_t H_t) P_{t|t-1}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, KellySizer, Backtester,
    Pair, compute_spread,
)


# =============================================================================
# §7 Kalman 동적 베타
# =============================================================================

def kalman_dynamic_beta(y_log: np.ndarray,
                        x_log: np.ndarray,
                        delta: float = 1e-5,
                        R: float = 1e-3,
                        warmup: int = 60) -> np.ndarray:
    """
    1차원 Kalman 필터로 동적 베타 추정.

    파라미터:
        y_log: 종속 자산 로그 가격 시계열
        x_log: 독립 자산 로그 가격 시계열
        delta: 베타 변동성 제어 (작을수록 베타 변화 느림)
        R: 관측 노이즈 분산
        warmup: 초기 베타 추정에 사용할 일수 (transient 회피)

    반환: 시변 베타 시계열 (길이 = len(y_log))

    수치 안정성:
        - warmup 기간은 OLS 베타로 고정
        - Q = delta / (1 - delta) ≈ delta (small delta)
        - 분모 0 방지 위해 H² · P + R 사용
    """
    n = len(y_log)
    beta = np.zeros(n)

    # Warmup: 초기 베타 OLS 추정
    if warmup < n and warmup > 5:
        beta_init = float(np.polyfit(x_log[:warmup], y_log[:warmup], 1)[0])
    else:
        beta_init = 1.0

    beta[:warmup] = beta_init
    P = 1.0
    Q = delta / (1 - delta)

    for t in range(warmup, n):
        # 예측 단계
        beta_pred = beta[t - 1]
        P_pred = P + Q

        # 갱신 단계
        H = x_log[t]
        K = P_pred * H / (H * H * P_pred + R)
        beta[t] = beta_pred + K * (y_log[t] - H * beta_pred)
        P = (1.0 - K * H) * P_pred

    return beta


def compute_dynamic_spread(price_y: pd.Series,
                           price_x: pd.Series,
                           delta: float = 1e-5,
                           R: float = 1e-3,
                           warmup: int = 60) -> tuple[pd.Series, pd.Series]:
    """
    동적 베타와 동적 spread를 함께 반환.

    반환:
        beta_series: 시변 베타 (pd.Series)
        spread_series: log(y) - β_t * log(x)
    """
    common = price_y.index.intersection(price_x.index)
    py = price_y.loc[common]
    px = price_x.loc[common]

    log_y = np.log(py.values)
    log_x = np.log(px.values)

    beta_arr = kalman_dynamic_beta(log_y, log_x, delta=delta, R=R, warmup=warmup)
    spread_arr = log_y - beta_arr * log_x

    beta_series = pd.Series(beta_arr, index=common, name='kalman_beta')
    spread_series = pd.Series(spread_arr, index=common, name='kalman_spread')
    return beta_series, spread_series


# =============================================================================
# Backtester for Dynamic Beta (페어 백테스트 수정 버전)
# =============================================================================

class DynamicBetaBacktester:
    """
    동적 베타에 맞춰 수정된 백테스터.

    원래 Backtester는 단일 베타로 페어 수익률을 계산했지만,
    동적 베타는 매일 변하므로 spread return을 직접 계산.
    """

    def __init__(self,
                 initial_capital: float = 100_000.0,
                 capital_fraction: float = 0.10,
                 fee_rate: float = 0.0004,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.capital_fraction = capital_fraction
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(self,
            price_y: pd.Series,
            price_x: pd.Series,
            beta_series: pd.Series,
            position: pd.Series,
            force_close: pd.Series):
        """
        delta neutral 가정에서 페어 수익률:
            ret_t = pos_{t-1} * (log_ret_y_t - β_{t-1} * log_ret_x_t) - cost_t

        β를 시변으로 사용하되, β_{t-1}을 곱해 look-ahead 차단.
        """
        df = pd.DataFrame({
            'py': price_y, 'px': price_x,
            'beta': beta_series,
            'pos': position.astype(int),
            'force': force_close.astype(bool),
        }).dropna()

        df['ry'] = np.log(df['py']).diff()
        df['rx'] = np.log(df['px']).diff()
        df['beta_prev'] = df['beta'].shift(1)

        df['pos_prev'] = df['pos'].shift(1).fillna(0).astype(int)
        df['trade'] = df['pos'] != df['pos_prev']
        cost_per_change = (self.fee_rate + self.slippage) * 2
        df['cost'] = df['trade'].astype(float) * cost_per_change

        # 동적 헤지된 페어 수익률
        df['strat_ret'] = (df['pos_prev'] *
                          (df['ry'] - df['beta_prev'] * df['rx']) -
                          df['cost'])

        df['daily_pnl_pct'] = (df['strat_ret'] * self.capital_fraction).fillna(0)
        df['equity'] = self.initial_capital * (1 + df['daily_pnl_pct']).cumprod()

        # 거래 추출 (Backtester와 동일 로직)
        trades = self._extract_trades(df)
        metrics = self._compute_metrics(df['equity'], df['daily_pnl_pct'], trades)

        # Backtester와 동일한 인터페이스로 반환
        from pairs_trading_mvp import BacktestResult
        return BacktestResult(
            equity=df['equity'],
            daily_pnl=df['daily_pnl_pct'] * self.initial_capital,
            position=df['pos'],
            trades=trades,
            metrics=metrics,
        )

    @staticmethod
    def _extract_trades(df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        in_position = False
        entry_date = None
        entry_pos = 0
        entry_equity = None

        for date, row in df.iterrows():
            if not in_position and row['pos'] != 0:
                in_position = True
                entry_date = date
                entry_pos = row['pos']
                entry_equity = row['equity']
            elif in_position and (row['pos'] == 0 or row['pos'] != entry_pos):
                trades.append({
                    'entry': entry_date,
                    'exit': date,
                    'side': entry_pos,
                    'duration': (date - entry_date).days,
                    'pnl_pct': (row['equity'] / entry_equity - 1)
                              if entry_equity else 0,
                })
                in_position = (row['pos'] != 0)
                if in_position:
                    entry_date = date
                    entry_pos = row['pos']
                    entry_equity = row['equity']
        return pd.DataFrame(trades)

    @staticmethod
    def _compute_metrics(equity: pd.Series,
                         daily_ret: pd.Series,
                         trades: pd.DataFrame) -> dict:
        if len(equity) < 2:
            return {}
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = ((1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
                if n_years > 0 else 0)
        ret = daily_ret.dropna()
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        cummax = equity.cummax()
        mdd = float((equity / cummax - 1).min())
        calmar = cagr / abs(mdd) if mdd < 0 else np.inf
        win_rate = (trades['pnl_pct'] > 0).mean() if len(trades) > 0 else 0
        avg_dur = trades['duration'].mean() if len(trades) > 0 else 0
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'calmar': calmar,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_duration': avg_dur,
        }


# =============================================================================
# 동적 베타 + K-fold 평가
# =============================================================================

def kfold_evaluate_pair_dynamic(price_y: pd.Series,
                                 price_x: pd.Series,
                                 pair: Pair,
                                 kfold,
                                 sig_window: int = 30,
                                 entry: float = 2.0,
                                 exit_thr: float = 0.5,
                                 stop: float = 3.5,
                                 kalman_delta: float = 1e-5,
                                 kalman_R: float = 1e-3,
                                 kalman_warmup: int = 60,
                                 kelly_fraction: float = 0.25,
                                 kelly_cap: float = 0.20,
                                 fee_rate: float = 0.0004,
                                 slippage: float = 0.0005,
                                 min_train_trades: int = 3):
    """동적 베타로 페어 K-fold 평가. 정적 버전과 시그니처 호환."""
    from purged_kfold import PairKFoldReport, FoldResult

    common = price_y.index.intersection(price_x.index)
    py = price_y.loc[common]
    px = price_x.loc[common]

    # 동적 베타 + spread
    beta_series, spread_series = compute_dynamic_spread(
        py, px, delta=kalman_delta, R=kalman_R, warmup=kalman_warmup)

    sig_gen = SignalGenerator(window=sig_window, entry=entry,
                              exit_thr=exit_thr, stop=stop)
    z, position, force_close = sig_gen.generate(spread_series)

    bt_full = DynamicBetaBacktester(initial_capital=100_000,
                                     capital_fraction=1.0,
                                     fee_rate=fee_rate,
                                     slippage=slippage)
    full_result = bt_full.run(py, px, beta_series, position, force_close)

    if full_result.trades.empty:
        return PairKFoldReport(pair=pair, fold_results=[], total_folds=0)

    trades = full_result.trades.copy()
    trades['entry'] = pd.to_datetime(trades['entry'])

    fold_results = []
    n = len(py)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(n)):
        train_dates = py.index[train_idx]
        val_dates = py.index[val_idx]

        train_trades = trades[trades['entry'].isin(train_dates)]
        val_trades = trades[trades['entry'].isin(val_dates)]

        if len(train_trades) < min_train_trades:
            continue

        kelly_f = KellySizer.from_trades(
            train_trades['pnl_pct'],
            fraction=kelly_fraction, cap=kelly_cap)

        train_sharpe = (
            train_trades['pnl_pct'].mean() / train_trades['pnl_pct'].std()
            * np.sqrt(252 / max(train_trades['duration'].mean(), 1))
            if train_trades['pnl_pct'].std() > 0 else 0
        )
        train_win_rate = (train_trades['pnl_pct'] > 0).mean()

        if len(val_trades) == 0:
            val_return = val_sharpe = val_mdd = val_win_rate = 0.0
        else:
            scaled_pnl = val_trades['pnl_pct'] * kelly_f
            cum = (1 + scaled_pnl).cumprod()
            val_return = float(cum.iloc[-1] - 1)
            val_sharpe = (
                scaled_pnl.mean() / scaled_pnl.std()
                * np.sqrt(252 / max(val_trades['duration'].mean(), 1))
                if scaled_pnl.std() > 0 else 0
            )
            running_max = cum.cummax()
            val_mdd = float((cum / running_max - 1).min())
            val_win_rate = (val_trades['pnl_pct'] > 0).mean()

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_n_trades=len(train_trades),
            train_win_rate=train_win_rate,
            train_sharpe=train_sharpe,
            kelly_f=kelly_f,
            val_n_trades=len(val_trades),
            val_win_rate=val_win_rate,
            val_total_return=val_return,
            val_sharpe=val_sharpe,
            val_max_drawdown=val_mdd,
        ))

    if not fold_results:
        return PairKFoldReport(pair=pair, fold_results=[], total_folds=0)

    val_returns = np.array([f.val_total_return for f in fold_results])
    val_sharpes = np.array([f.val_sharpe for f in fold_results])

    return PairKFoldReport(
        pair=pair,
        fold_results=fold_results,
        mean_val_return=float(val_returns.mean()),
        std_val_return=float(val_returns.std()),
        mean_val_sharpe=float(val_sharpes.mean()),
        std_val_sharpe=float(val_sharpes.std()),
        robustness_score=(float(val_sharpes.mean() / val_sharpes.std())
                         if val_sharpes.std() > 1e-6 else 0),
        profitable_folds=int((val_returns > 0).sum()),
        total_folds=len(fold_results),
    )


if __name__ == "__main__":
    print("kalman_pairs module loaded.")
    print("  kalman_dynamic_beta, compute_dynamic_spread, "
          "DynamicBetaBacktester, kfold_evaluate_pair_dynamic")
