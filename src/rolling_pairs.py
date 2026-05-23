"""
Rolling Pairs — 시간에 따라 페어 풀을 자동 갱신
=================================================

근본 문제 (5차 MVP까지의 결론):
  - 1~5차 MVP 모두 같은 페어 풀에서 베타 추정 방식만 다름
  - 페어 자체가 죽어가는 문제(regime change)는 어떤 베타 추정도 해결 못 함
  - GDX~KO의 경우 2025년 7월 금값 급등으로 KO와 평소 관계 깨짐

이 모듈의 해결책:
  - 매 N일마다 (예: 60일) 전체 페어 풀을 재검정
  - 죽은 페어 자동 폐기, 새 페어 자동 채택
  - 살아남은 페어는 베타와 half-life 갱신

비유로 풀면:
  - 1~5차: "이 페어 한 번 결정하면 끝까지 가자"
  - 6차: "정기 건강검진. 죽은 페어는 폐기, 새 페어는 영입"
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import PairsFinder, Pair


@dataclass
class PoolSnapshot:
    """특정 시점의 페어 풀 스냅샷"""
    date: pd.Timestamp
    pairs: list[Pair]
    n_pairs: int = 0
    new_pairs: list[str] = field(default_factory=list)      # 신규 영입
    dropped_pairs: list[str] = field(default_factory=list)  # 폐기
    survived_pairs: list[str] = field(default_factory=list) # 유지

    def __post_init__(self):
        self.n_pairs = len(self.pairs)


class RollingPairsManager:
    """
    시간 따라 페어 풀을 자동 갱신.

    파라미터:
        finder: PairsFinder 인스턴스
        lookback_days: 페어 검정에 사용할 과거 데이터 일수
        refresh_every_days: 페어 풀 재검정 주기

    동작:
        1. 시점 t에서 [t-lookback, t] 데이터로 페어 발굴
        2. 발굴된 페어 풀을 t에 기록
        3. t + refresh_every 시점에 1번 반복
    """

    def __init__(self,
                 finder: PairsFinder,
                 lookback_days: int = 365,
                 refresh_every_days: int = 60):
        self.finder = finder
        self.lookback_days = lookback_days
        self.refresh_every_days = refresh_every_days
        self.snapshots: list[PoolSnapshot] = []

    def run(self, price_df: pd.DataFrame,
            start_offset_days: Optional[int] = None,
            verbose: bool = False) -> list[PoolSnapshot]:
        """
        전체 데이터를 시간 순으로 훑으며 페어 풀 갱신.

        반환: 시점별 페어 풀 스냅샷 리스트.
        """
        if start_offset_days is None:
            start_offset_days = self.lookback_days

        dates = price_df.index
        n = len(dates)
        self.snapshots = []
        prev_pair_ids = set()

        for i in range(start_offset_days, n, self.refresh_every_days):
            current_date = dates[i]
            window_start = max(0, i - self.lookback_days)
            window_df = price_df.iloc[window_start:i]

            try:
                pairs = self.finder.screen_pairs(window_df)
            except Exception:
                pairs = []

            # 페어 ID로 변동 분석
            current_pair_ids = {f'{p.y}~{p.x}' for p in pairs}
            new_ids = current_pair_ids - prev_pair_ids
            dropped_ids = prev_pair_ids - current_pair_ids
            survived_ids = current_pair_ids & prev_pair_ids

            snapshot = PoolSnapshot(
                date=current_date,
                pairs=pairs,
                new_pairs=sorted(new_ids),
                dropped_pairs=sorted(dropped_ids),
                survived_pairs=sorted(survived_ids),
            )
            self.snapshots.append(snapshot)

            if verbose:
                print(f'  {current_date.date()}: '
                      f'{len(pairs)} pairs ('
                      f'+{len(new_ids)} new, '
                      f'-{len(dropped_ids)} dropped, '
                      f'={len(survived_ids)} survived)')

            prev_pair_ids = current_pair_ids

        return self.snapshots

    def get_lifecycle(self) -> pd.DataFrame:
        """
        각 페어가 어느 시점에 적격이었는지 추적.

        반환: DataFrame (index=date, columns=pair_id, values=boolean)
                True = 적격, False = 부적격
        """
        all_pair_ids = set()
        for s in self.snapshots:
            for p in s.pairs:
                all_pair_ids.add(f'{p.y}~{p.x}')

        dates = [s.date for s in self.snapshots]
        df = pd.DataFrame(False, index=dates, columns=sorted(all_pair_ids))

        for s in self.snapshots:
            for p in s.pairs:
                df.loc[s.date, f'{p.y}~{p.x}'] = True

        return df

    def summary(self) -> dict:
        """요약 통계."""
        if not self.snapshots:
            return {}

        all_pair_ids = set()
        for s in self.snapshots:
            for p in s.pairs:
                all_pair_ids.add(f'{p.y}~{p.x}')

        # 각 페어가 몇 번 적격이었나
        lifecycle = self.get_lifecycle()
        appearance_counts = lifecycle.sum()

        # 한 번이라도 적격이었던 페어
        ever_qualified = len(all_pair_ids)
        # 모든 시점에서 적격
        always_qualified = int((appearance_counts == len(self.snapshots)).sum())
        # 한 번만 적격 (한 번 떴다가 사라짐)
        only_once = int((appearance_counts == 1).sum())

        avg_pool_size = np.mean([s.n_pairs for s in self.snapshots])
        avg_new_per_refresh = np.mean([len(s.new_pairs) for s in self.snapshots[1:]])
        avg_dropped_per_refresh = np.mean([len(s.dropped_pairs)
                                            for s in self.snapshots[1:]])

        return {
            'n_snapshots': len(self.snapshots),
            'ever_qualified_pairs': ever_qualified,
            'always_qualified': always_qualified,
            'only_once': only_once,
            'avg_pool_size': avg_pool_size,
            'avg_new_per_refresh': avg_new_per_refresh,
            'avg_dropped_per_refresh': avg_dropped_per_refresh,
        }


# =============================================================================
# 단순 Walk-Forward 백테스트 (단일 페어 운용)
# =============================================================================

@dataclass
class WalkForwardResult:
    pair_lifecycle: pd.DataFrame   # 페어별 적격 여부 시계열
    snapshots: list[PoolSnapshot]
    summary_stats: dict
    pair_durations: pd.Series      # 각 페어의 총 적격 일수


def walk_forward_pair_lifecycle(
        price_df: pd.DataFrame,
        finder: PairsFinder,
        lookback_days: int = 365,
        refresh_every_days: int = 60,
        verbose: bool = False) -> WalkForwardResult:
    """
    Walk-forward로 페어 풀의 생애주기 추적.

    가장 간단한 형태의 walk-forward — 백테스트는 하지 않고
    페어 풀이 시간에 따라 어떻게 변하는지만 본다.

    핵심 질문:
      1. GDX~KO는 언제 페어 풀에서 빠지는가?
      2. 새 페어가 그 자리를 채우는가?
      3. 평균적으로 페어 수가 어떻게 변하는가?
    """
    manager = RollingPairsManager(
        finder=finder,
        lookback_days=lookback_days,
        refresh_every_days=refresh_every_days,
    )
    snapshots = manager.run(price_df, verbose=verbose)
    lifecycle = manager.get_lifecycle()
    summary_stats = manager.summary()

    # 각 페어의 총 적격 일수 (각 시점이 refresh_every_days를 대표)
    pair_durations = (lifecycle.sum() * refresh_every_days).sort_values(
        ascending=False)

    return WalkForwardResult(
        pair_lifecycle=lifecycle,
        snapshots=snapshots,
        summary_stats=summary_stats,
        pair_durations=pair_durations,
    )


if __name__ == "__main__":
    print("rolling_pairs module loaded.")
    print("  RollingPairsManager, walk_forward_pair_lifecycle")
