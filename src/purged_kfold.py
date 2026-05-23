"""
Purged K-fold 검증 프레임워크 — 명세서 v2 §8
=============================================

정통 López de Prado 구현:
  - Purge: 검증셋 시작 시점 이전 purge_days만큼 훈련셋에서 제거
  - Embargo: 검증셋 끝난 직후 embargo 비율만큼 훈련셋에서 제거

페어 트레이딩 적용:
  - 신호는 전체 시계열에 대해 한 번 계산 (시간 순 안전)
  - 거래 로그를 fold별로 train/val 분할 (entry 날짜 기준)
  - train 거래로 켈리 추정 → val 거래에 적용
  - K개 fold의 OOS 수익률 분포로 견고성 판단
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator, Optional
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, KellySizer, Backtester,
    Pair, compute_spread,
)


# =============================================================================
# §8  PurgedKFold
# =============================================================================

class PurgedKFold:
    """López de Prado 정통 구현.

    Purge: val 시작 이전 purge_days 훈련셋에서 제거 (라벨 leakage 차단)
    Embargo: val 끝난 직후 embargo 훈련셋에서 제거 (자기상관 차단)
    """

    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 30,
                 embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_pct = embargo_pct

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """반환: (train_idx, val_idx) generator"""
        fold_size = n // self.n_splits
        embargo = int(n * self.embargo_pct)

        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < self.n_splits - 1 else n

            train_mask = np.ones(n, dtype=bool)
            # Purge: 검증셋 앞쪽 (검증셋 자체 포함)
            train_mask[max(0, val_start - self.purge_days):val_end] = False
            # Embargo: 검증셋 직후
            train_mask[val_end:min(n, val_end + embargo)] = False

            train_idx = np.where(train_mask)[0]
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx


# =============================================================================
# 페어 트레이딩 K-fold 평가
# =============================================================================

@dataclass
class FoldResult:
    fold_idx: int
    train_n_trades: int
    train_win_rate: float
    train_sharpe: float
    kelly_f: float
    val_n_trades: int
    val_win_rate: float
    val_total_return: float
    val_sharpe: float
    val_max_drawdown: float


@dataclass
class PairKFoldReport:
    pair: Pair
    fold_results: list[FoldResult]
    # 집계 통계
    mean_val_return: float = 0.0
    std_val_return: float = 0.0
    mean_val_sharpe: float = 0.0
    std_val_sharpe: float = 0.0
    robustness_score: float = 0.0  # mean / std (높을수록 견고)
    profitable_folds: int = 0
    total_folds: int = 0

    def summary(self) -> str:
        return (
            f"{self.pair.y}~{self.pair.x}: "
            f"OOS μ={self.mean_val_return:+.2%}±{self.std_val_return:.2%} "
            f"Sharpe μ={self.mean_val_sharpe:+.2f}±{self.std_val_sharpe:.2f} "
            f"수익fold={self.profitable_folds}/{self.total_folds} "
            f"견고성={self.robustness_score:+.2f}"
        )


def kfold_evaluate_pair(
        price_y: pd.Series,
        price_x: pd.Series,
        pair: Pair,
        kfold: PurgedKFold,
        sig_window: int = 30,
        entry: float = 2.0,
        exit_thr: float = 0.5,
        stop: float = 3.5,
        kelly_fraction: float = 0.25,
        kelly_cap: float = 0.20,
        fee_rate: float = 0.0004,
        slippage: float = 0.0005,
        min_train_trades: int = 3,
) -> PairKFoldReport:
    """단일 페어에 대해 K-fold 평가."""

    # 전체 시계열 정렬
    common_idx = price_y.index.intersection(price_x.index)
    py = price_y.loc[common_idx]
    px = price_x.loc[common_idx]

    # 전체 신호 생성 (시간 순 안전)
    spread = compute_spread(py, px, pair.beta)
    sig_gen = SignalGenerator(window=sig_window, entry=entry,
                              exit_thr=exit_thr, stop=stop)
    z, position, force_close = sig_gen.generate(spread)

    # 전체 백테스트 (capital_fraction=1.0으로 raw 거래 추출용)
    bt_full = Backtester(initial_capital=100_000,
                         capital_fraction=1.0,
                         fee_rate=fee_rate,
                         slippage=slippage)
    full_result = bt_full.run(py, px, pair.beta, position, force_close)

    if full_result.trades.empty:
        return PairKFoldReport(pair=pair, fold_results=[], total_folds=0)

    trades = full_result.trades.copy()
    trades['entry'] = pd.to_datetime(trades['entry'])

    # K-fold 평가
    fold_results = []
    n = len(py)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(n)):
        train_dates = py.index[train_idx]
        val_dates = py.index[val_idx]

        # 거래 분할: entry 날짜 기준
        train_trades = trades[trades['entry'].isin(train_dates)]
        val_trades = trades[trades['entry'].isin(val_dates)]

        # train 켈리 추정
        if len(train_trades) < min_train_trades:
            continue

        kelly_f = KellySizer.from_trades(
            train_trades['pnl_pct'],
            fraction=kelly_fraction,
            cap=kelly_cap,
        )

        # train 통계
        train_sharpe = (
            train_trades['pnl_pct'].mean() / train_trades['pnl_pct'].std()
            * np.sqrt(252 / max(train_trades['duration'].mean(), 1))
            if train_trades['pnl_pct'].std() > 0 else 0
        )
        train_win_rate = (train_trades['pnl_pct'] > 0).mean()

        # val 평가: 켈리 적용한 PnL
        if len(val_trades) == 0:
            val_return = 0.0
            val_sharpe = 0.0
            val_mdd = 0.0
            val_win_rate = 0.0
        else:
            scaled_pnl = val_trades['pnl_pct'] * kelly_f
            # 복리 합산
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

    # 집계
    if not fold_results:
        return PairKFoldReport(pair=pair, fold_results=[], total_folds=0)

    val_returns = np.array([f.val_total_return for f in fold_results])
    val_sharpes = np.array([f.val_sharpe for f in fold_results])

    mean_ret = float(val_returns.mean())
    std_ret = float(val_returns.std())
    mean_sh = float(val_sharpes.mean())
    std_sh = float(val_sharpes.std())

    # 견고성 점수: Information Ratio 스타일
    robustness = mean_sh / std_sh if std_sh > 1e-6 else 0
    profitable = int((val_returns > 0).sum())

    return PairKFoldReport(
        pair=pair,
        fold_results=fold_results,
        mean_val_return=mean_ret,
        std_val_return=std_ret,
        mean_val_sharpe=mean_sh,
        std_val_sharpe=std_sh,
        robustness_score=robustness,
        profitable_folds=profitable,
        total_folds=len(fold_results),
    )


if __name__ == "__main__":
    print("purged_kfold module loaded.")
    print("  PurgedKFold, kfold_evaluate_pair, PairKFoldReport")
