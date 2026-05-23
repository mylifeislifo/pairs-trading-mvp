"""
Adaptive Dual-Strategy Ensemble — 5차 MVP
==========================================

4차 MVP의 발견:
  - 정적 베타: 평소(fold 1, 2)에서 강함
  - Kalman 베타: 시장 격변(fold 3)에서 강함
  - 평균적으로 정적이 우수하지만 사각지대 존재

이 모듈의 가설:
  "두 전략을 동시에 운용하고, 최근 성과에 따라 자본을 자동 배분하면
   각 시장 국면에서 잘 작동하는 쪽이 우세해진다."

가중치 알고리즘 — "최근 성과 기반 적응형 가중치":
  1. 최근 N일(rolling lookback) 동안 두 전략의 일별 수익률 계산
  2. 두 전략의 Sharpe 비율 측정
  3. 더 좋은 쪽에 더 많은 자본 배분 (softmax 가중)
  4. 매일 가중치 업데이트

비유로 설명:
  - 두 명의 펀드매니저를 고용
  - 한 명은 안정 시장 전문 (정적)
  - 다른 한 명은 격변 시장 전문 (Kalman)
  - 매일 누가 최근에 더 잘했는지 보고 자본을 그쪽으로 옮김
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
from purged_kfold import PurgedKFold, PairKFoldReport, FoldResult
from kalman_pairs import (
    compute_dynamic_spread, DynamicBetaBacktester,
)


# =============================================================================
# 적응형 가중치 계산
# =============================================================================

def adaptive_weight(
        ret_static: pd.Series,
        ret_dynamic: pd.Series,
        lookback: int = 30,
        temperature: float = 5.0,
        min_weight: float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """
    최근 성과 기반 적응형 가중치 (softmax).

    파라미터:
        ret_static, ret_dynamic: 두 전략의 일별 수익률
        lookback: 가중치 계산용 윈도우 (일)
        temperature: 가중치 민감도 (높을수록 최근 잘하는 쪽에 쏠림)
        min_weight: 한쪽의 최소 가중치 (0~1, 완전 쏠림 방지)

    반환:
        w_static, w_dynamic: 가중치 시계열 (합 = 1)

    수치 안정성:
        - lookback 이전엔 50:50 균등
        - sigma=0이면 균등으로 fallback
        - look-ahead 차단을 위해 shift(1) 적용
    """
    # 롤링 Sharpe (look-ahead 차단)
    def rolling_sharpe(ret, w):
        m = ret.rolling(w).mean().shift(1)
        s = ret.rolling(w).std().shift(1)
        return (m / s.replace(0, np.nan)).fillna(0)

    sr_s = rolling_sharpe(ret_static, lookback)
    sr_d = rolling_sharpe(ret_dynamic, lookback)

    # Softmax 가중치
    exp_s = np.exp(temperature * sr_s)
    exp_d = np.exp(temperature * sr_d)
    denom = (exp_s + exp_d).replace(0, np.nan)

    w_s = (exp_s / denom).fillna(0.5)
    w_d = (exp_d / denom).fillna(0.5)

    # 최소 가중치 clip (완전 쏠림 방지)
    w_s = w_s.clip(lower=min_weight, upper=1 - min_weight)
    w_d = 1 - w_s

    return w_s, w_d


# =============================================================================
# Ensemble Backtester
# =============================================================================

@dataclass
class EnsembleResult:
    equity: pd.Series
    daily_pnl: pd.Series
    weights_static: pd.Series  # 가중치 추이 (모니터링용)
    weights_dynamic: pd.Series
    static_daily_ret: pd.Series  # 두 전략의 일별 수익률
    dynamic_daily_ret: pd.Series
    metrics: dict


class EnsembleBacktester:
    """
    두 전략(정적/동적)을 적응형 가중치로 합성.

    작동 방식:
      1. 정적 백테스트 → 일별 수익률 시계열
      2. 동적 백테스트 → 일별 수익률 시계열
      3. 매일 두 전략의 최근 성과로 가중치 계산
      4. 합성 수익률 = w_s × ret_s + w_d × ret_d
      5. 합성 equity 곡선 생성
    """

    def __init__(self,
                 initial_capital: float = 100_000.0,
                 capital_fraction: float = 0.10,
                 fee_rate: float = 0.0004,
                 slippage: float = 0.0005,
                 lookback: int = 30,
                 temperature: float = 5.0,
                 min_weight: float = 0.1):
        self.initial_capital = initial_capital
        self.capital_fraction = capital_fraction
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.lookback = lookback
        self.temperature = temperature
        self.min_weight = min_weight

    def run(self,
            price_y: pd.Series,
            price_x: pd.Series,
            pair: Pair,
            sig_window: int = 30,
            entry: float = 2.0,
            exit_thr: float = 0.5,
            stop: float = 3.5,
            kalman_delta: float = 1e-5,
            kalman_R: float = 1e-3,
            kalman_warmup: int = 60) -> EnsembleResult:

        # 1. 정적 베타 전략
        static_spread = compute_spread(price_y, price_x, pair.beta)
        sig_gen = SignalGenerator(window=sig_window, entry=entry,
                                  exit_thr=exit_thr, stop=stop)
        z_s, pos_s, fc_s = sig_gen.generate(static_spread)

        bt_static = Backtester(initial_capital=self.initial_capital,
                              capital_fraction=1.0,
                              fee_rate=self.fee_rate,
                              slippage=self.slippage)
        res_static = bt_static.run(price_y, price_x, pair.beta, pos_s, fc_s)

        # 2. 동적 베타 전략
        beta_dyn, spread_dyn = compute_dynamic_spread(
            price_y, price_x, delta=kalman_delta, R=kalman_R,
            warmup=kalman_warmup)
        z_d, pos_d, fc_d = sig_gen.generate(spread_dyn)

        bt_dynamic = DynamicBetaBacktester(initial_capital=self.initial_capital,
                                            capital_fraction=1.0,
                                            fee_rate=self.fee_rate,
                                            slippage=self.slippage)
        res_dynamic = bt_dynamic.run(price_y, price_x, beta_dyn, pos_d, fc_d)

        # 3. 두 전략의 일별 수익률 (로그수익률 형태)
        ret_static = res_static.equity.pct_change().fillna(0)
        ret_dynamic = res_dynamic.equity.pct_change().fillna(0)

        # 인덱스 정렬
        common_idx = ret_static.index.intersection(ret_dynamic.index)
        ret_static = ret_static.loc[common_idx]
        ret_dynamic = ret_dynamic.loc[common_idx]

        # 4. 적응형 가중치
        w_s, w_d = adaptive_weight(ret_static, ret_dynamic,
                                    lookback=self.lookback,
                                    temperature=self.temperature,
                                    min_weight=self.min_weight)

        # 5. 합성 수익률 (전일 가중치 × 당일 수익률 - look-ahead 차단)
        ensemble_ret = (w_s.shift(1).fillna(0.5) * ret_static +
                       w_d.shift(1).fillna(0.5) * ret_dynamic)

        # capital_fraction 적용
        daily_pnl_pct = ensemble_ret * self.capital_fraction
        equity = self.initial_capital * (1 + daily_pnl_pct).cumprod()

        # 6. 성능 지표
        metrics = self._compute_metrics(equity, daily_pnl_pct)
        metrics['static_sharpe'] = res_static.metrics.get('sharpe', 0)
        metrics['dynamic_sharpe'] = res_dynamic.metrics.get('sharpe', 0)
        metrics['static_return'] = res_static.metrics.get('total_return', 0)
        metrics['dynamic_return'] = res_dynamic.metrics.get('total_return', 0)
        metrics['avg_static_weight'] = float(w_s.mean())

        return EnsembleResult(
            equity=equity,
            daily_pnl=daily_pnl_pct * self.initial_capital,
            weights_static=w_s,
            weights_dynamic=w_d,
            static_daily_ret=ret_static,
            dynamic_daily_ret=ret_dynamic,
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(equity: pd.Series, daily_ret: pd.Series) -> dict:
        if len(equity) < 2:
            return {}
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = ((1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
                if n_years > 0 else 0)
        ret = daily_ret.dropna()
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) \
                 if ret.std() > 0 else 0
        cummax = equity.cummax()
        mdd = float((equity / cummax - 1).min())
        calmar = cagr / abs(mdd) if mdd < 0 else np.inf
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'calmar': calmar,
        }


# =============================================================================
# K-fold 평가 (Ensemble)
# =============================================================================

def kfold_evaluate_ensemble(
        price_y: pd.Series,
        price_x: pd.Series,
        pair: Pair,
        kfold: PurgedKFold,
        sig_window: int = 30,
        lookback: int = 30,
        temperature: float = 5.0,
        min_weight: float = 0.1,
        kalman_delta: float = 1e-5,
) -> PairKFoldReport:
    """Ensemble 전략의 K-fold 평가. 정적·동적 버전과 시그니처 호환."""
    common = price_y.index.intersection(price_x.index)
    py = price_y.loc[common]
    px = price_x.loc[common]

    # 전체 백테스트 (capital_fraction=1.0)
    bt = EnsembleBacktester(initial_capital=100_000,
                            capital_fraction=1.0,
                            lookback=lookback,
                            temperature=temperature,
                            min_weight=min_weight)
    res = bt.run(py, px, pair, sig_window=sig_window,
                 kalman_delta=kalman_delta)

    if len(res.equity) < 10:
        return PairKFoldReport(pair=pair, fold_results=[], total_folds=0)

    # 일별 수익률을 fold별로 분할
    daily_ret = res.daily_pnl / 100_000  # 비율로
    daily_ret.index = pd.to_datetime(daily_ret.index)

    fold_results = []
    n = len(daily_ret)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(n)):
        train_ret = daily_ret.iloc[train_idx]
        val_ret = daily_ret.iloc[val_idx]

        # Train 통계로 Kelly 추정 (간단화: Sharpe 기반)
        if train_ret.std() == 0:
            continue
        train_sharpe = float(train_ret.mean() / train_ret.std() * np.sqrt(252))

        # Kelly: 단순 Sharpe → 가중치 (근사적)
        kelly_f = max(0, min(0.2, train_sharpe / 10))

        if val_ret.std() == 0:
            val_sharpe = 0
            val_return = 0
            val_mdd = 0
        else:
            scaled_val = val_ret * kelly_f / max(daily_ret.std(), 1e-6)
            val_sharpe = float(scaled_val.mean() / scaled_val.std()
                              * np.sqrt(252)) if scaled_val.std() > 0 else 0
            val_return = float((1 + scaled_val).cumprod().iloc[-1] - 1)
            cum = (1 + scaled_val).cumprod()
            val_mdd = float((cum / cum.cummax() - 1).min())

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_n_trades=int(train_ret.abs().sum() > 0),
            train_win_rate=float((train_ret > 0).mean()),
            train_sharpe=train_sharpe,
            kelly_f=kelly_f,
            val_n_trades=int(val_ret.abs().sum() > 0),
            val_win_rate=float((val_ret > 0).mean()),
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
    print("ensemble_strategy module loaded.")
    print("  adaptive_weight, EnsembleBacktester, kfold_evaluate_ensemble")
