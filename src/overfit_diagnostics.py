"""
Overfitting Diagnostics — 오버핏 정량 진단
==========================================

두 가지 핵심 진단 도구:

1. ParameterSensitivityAnalyzer:
   - (window, entry, exit_thr) 그리드 탐색
   - 각 조합에서 K-fold 평균 OOS Sharpe 측정
   - 진짜 견고한 페어 = heatmap이 대체로 균일하게 좋음
   - Overfit 페어 = 좁은 sweet spot만 존재

2. PermutationTester:
   - position 신호를 시계열 무작위 셔플
   - B번 반복하여 셔플 Sharpe 분포 생성
   - 실제 Sharpe의 통계적 유의성 p-value 계산
   - p < 0.05면 진짜 신호, p >= 0.05면 우연/오버핏

추가 도구:
   - OverfitScorer: 종합 점수 (낮을수록 오버핏 위험 작음)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import product
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    SignalGenerator, KellySizer, Backtester, Pair, compute_spread
)
from purged_kfold import PurgedKFold


# =============================================================================
# 도구 1: 파라미터 민감도 분석
# =============================================================================

@dataclass
class SensitivityResult:
    pair: Pair
    # 파라미터 격자: (window, entry) → 평균 OOS Sharpe
    sharpe_grid: pd.DataFrame
    # 격자 통계
    mean_sharpe: float = 0.0       # 모든 셀의 평균
    std_sharpe: float = 0.0        # 셀 간 변동성 (낮을수록 견고)
    positive_pct: float = 0.0      # Sharpe > 0인 셀 비율
    max_sharpe: float = 0.0        # 최고 셀
    min_sharpe: float = 0.0        # 최저 셀
    # 견고성 점수: mean / std (높을수록 균일하게 좋음)
    robustness: float = 0.0
    # 오버핏 신호: max - mean (특정 셀만 튀면 오버핏)
    overfit_signal: float = 0.0


class ParameterSensitivityAnalyzer:
    """파라미터 그리드 탐색으로 오버핏 진단."""

    def __init__(self,
                 windows: list[int] = None,
                 entries: list[float] = None,
                 exits: list[float] = None,
                 stop: float = 3.5,
                 kfold: Optional[PurgedKFold] = None):
        self.windows = windows or [20, 30, 40]
        self.entries = entries or [1.5, 2.0, 2.5]
        self.exits = exits or [0.0, 0.5]
        self.stop = stop
        self.kfold = kfold or PurgedKFold(n_splits=5, purge_days=30,
                                            embargo_pct=0.01)

    def analyze(self,
                price_y: pd.Series,
                price_x: pd.Series,
                pair: Pair) -> SensitivityResult:
        """페어에 대해 파라미터 격자 탐색."""
        # window × entry 그리드 (exit는 별도 처리)
        # 단순화 위해 exit는 entry/4로 고정
        grid = pd.DataFrame(index=self.windows, columns=self.entries,
                            dtype=float)
        grid.index.name = 'window'
        grid.columns.name = 'entry'

        for w, e in product(self.windows, self.entries):
            try:
                sharpe = self._evaluate_combo(price_y, price_x, pair,
                                              w, e, e * 0.25)
                grid.loc[w, e] = sharpe
            except Exception:
                grid.loc[w, e] = np.nan

        # 통계
        valid = grid.values[~np.isnan(grid.values)]
        if len(valid) == 0:
            return SensitivityResult(pair=pair, sharpe_grid=grid)

        mean_s = float(valid.mean())
        std_s = float(valid.std())
        pos_pct = float((valid > 0).mean())
        max_s = float(valid.max())
        min_s = float(valid.min())
        robust = mean_s / std_s if std_s > 1e-6 else 0.0
        overfit_sig = max_s - mean_s

        return SensitivityResult(
            pair=pair,
            sharpe_grid=grid,
            mean_sharpe=mean_s,
            std_sharpe=std_s,
            positive_pct=pos_pct,
            max_sharpe=max_s,
            min_sharpe=min_s,
            robustness=robust,
            overfit_signal=overfit_sig,
        )

    def _evaluate_combo(self,
                        price_y: pd.Series,
                        price_x: pd.Series,
                        pair: Pair,
                        window: int,
                        entry: float,
                        exit_thr: float) -> float:
        """단일 파라미터 조합에 대해 K-fold 평균 OOS Sharpe."""
        common = price_y.index.intersection(price_x.index)
        py = price_y.loc[common]
        px = price_x.loc[common]

        spread = compute_spread(py, px, pair.beta)
        sig_gen = SignalGenerator(window=window, entry=entry,
                                  exit_thr=exit_thr, stop=self.stop)
        z, position, force_close = sig_gen.generate(spread)

        bt = Backtester(initial_capital=100_000, capital_fraction=1.0)
        full_result = bt.run(py, px, pair.beta, position, force_close)

        if full_result.trades.empty:
            return 0.0

        trades = full_result.trades.copy()
        trades['entry'] = pd.to_datetime(trades['entry'])

        # K-fold 평균
        fold_sharpes = []
        n = len(py)
        for train_idx, val_idx in self.kfold.split(n):
            val_dates = py.index[val_idx]
            val_trades = trades[trades['entry'].isin(val_dates)]
            if len(val_trades) < 2:
                continue
            pnl = val_trades['pnl_pct']
            sh = pnl.mean() / pnl.std() * np.sqrt(252 / max(
                val_trades['duration'].mean(), 1))
            if pnl.std() > 0:
                fold_sharpes.append(sh)

        return float(np.mean(fold_sharpes)) if fold_sharpes else 0.0


# =============================================================================
# 도구 2: Permutation Test
# =============================================================================

@dataclass
class PermutationResult:
    pair: Pair
    real_sharpe: float
    null_distribution: np.ndarray  # 셔플 Sharpe 분포
    p_value: float
    null_mean: float
    null_std: float
    z_score: float  # (real - null_mean) / null_std

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


class PermutationTester:
    """position 신호를 무작위 셔플하여 진짜 신호 vs 노이즈 구분."""

    def __init__(self,
                 n_permutations: int = 200,
                 sig_window: int = 30,
                 entry: float = 2.0,
                 exit_thr: float = 0.5,
                 stop: float = 3.5,
                 random_state: int = 42):
        self.n_permutations = n_permutations
        self.sig_window = sig_window
        self.entry = entry
        self.exit_thr = exit_thr
        self.stop = stop
        self.rng = np.random.default_rng(random_state)

    def test(self,
             price_y: pd.Series,
             price_x: pd.Series,
             pair: Pair) -> PermutationResult:
        """페어에 대해 permutation test 실행."""
        common = price_y.index.intersection(price_x.index)
        py = price_y.loc[common]
        px = price_x.loc[common]

        # 실제 신호
        spread = compute_spread(py, px, pair.beta)
        sig_gen = SignalGenerator(window=self.sig_window, entry=self.entry,
                                  exit_thr=self.exit_thr, stop=self.stop)
        z, position, force_close = sig_gen.generate(spread)

        # 실제 Sharpe
        real_sharpe = self._compute_strategy_sharpe(
            py, px, pair.beta, position, force_close)

        # Null 분포: position을 무작위 셔플
        null_sharpes = []
        position_values = position.values
        force_close_values = force_close.values

        for _ in range(self.n_permutations):
            shuffled_idx = self.rng.permutation(len(position_values))
            shuffled_position = pd.Series(
                position_values[shuffled_idx], index=position.index)
            shuffled_force = pd.Series(
                force_close_values[shuffled_idx], index=force_close.index)
            null_sh = self._compute_strategy_sharpe(
                py, px, pair.beta, shuffled_position, shuffled_force)
            null_sharpes.append(null_sh)

        null_arr = np.array(null_sharpes)
        # p-value: 실제 Sharpe가 null 분포의 상위 몇 %인가
        p_value = float((null_arr >= real_sharpe).mean())
        null_mean = float(null_arr.mean())
        null_std = float(null_arr.std())
        z_score = ((real_sharpe - null_mean) / null_std
                   if null_std > 1e-6 else 0.0)

        return PermutationResult(
            pair=pair,
            real_sharpe=real_sharpe,
            null_distribution=null_arr,
            p_value=p_value,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
        )

    @staticmethod
    def _compute_strategy_sharpe(py, px, beta, position, force_close) -> float:
        """주어진 position 시계열의 전략 Sharpe."""
        bt = Backtester(initial_capital=100_000, capital_fraction=1.0)
        result = bt.run(py, px, beta, position, force_close)
        return float(result.metrics.get('sharpe', 0))


# =============================================================================
# 종합 오버핏 점수
# =============================================================================

@dataclass
class OverfitReport:
    pair: Pair
    # 진단 결과
    sensitivity: SensitivityResult
    permutation: PermutationResult
    # 종합 점수
    is_robust: bool = False
    is_significant: bool = False
    overall_grade: str = 'F'  # A/B/C/D/F

    def summary(self) -> str:
        return (
            f"{self.pair.y}~{self.pair.x}: "
            f"Sensitivity={self.sensitivity.mean_sharpe:+.2f}±"
            f"{self.sensitivity.std_sharpe:.2f} "
            f"(+%={self.sensitivity.positive_pct:.0%}), "
            f"Perm p={self.permutation.p_value:.3f} "
            f"(z={self.permutation.z_score:+.2f}), "
            f"Grade={self.overall_grade}"
        )


def grade_pair(sensitivity: SensitivityResult,
               permutation: PermutationResult) -> tuple[str, bool, bool]:
    """4가지 기준으로 페어 등급 매김.
    A: 모든 기준 통과 (진짜 견고)
    B: 3개 통과
    C: 2개 통과
    D: 1개 통과
    F: 전부 실패 (오버핏 의심)
    """
    criteria = {
        'sensitivity_positive': sensitivity.positive_pct >= 0.7,  # 70% 셀이 양수
        'sensitivity_robust': sensitivity.robustness >= 0.5,       # 견고성 0.5+
        'perm_significant': permutation.p_value < 0.10,            # 통계적 유의
        'perm_zscore': permutation.z_score >= 1.0,                  # 1σ 이상
    }
    n_pass = sum(criteria.values())

    grade_map = {4: 'A', 3: 'B', 2: 'C', 1: 'D', 0: 'F'}
    grade = grade_map[n_pass]
    is_robust = n_pass >= 3
    is_sig = criteria['perm_significant']
    return grade, is_robust, is_sig


def diagnose_pair(price_y: pd.Series,
                  price_x: pd.Series,
                  pair: Pair,
                  analyzer: ParameterSensitivityAnalyzer,
                  tester: PermutationTester) -> OverfitReport:
    """페어 종합 진단."""
    sens = analyzer.analyze(price_y, price_x, pair)
    perm = tester.test(price_y, price_x, pair)
    grade, robust, sig = grade_pair(sens, perm)

    return OverfitReport(
        pair=pair,
        sensitivity=sens,
        permutation=perm,
        is_robust=robust,
        is_significant=sig,
        overall_grade=grade,
    )


if __name__ == "__main__":
    print("overfit_diagnostics module loaded.")
    print("  ParameterSensitivityAnalyzer, PermutationTester, diagnose_pair")
