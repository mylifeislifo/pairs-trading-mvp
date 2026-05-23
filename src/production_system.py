"""
Production Pairs Trading System — 7차 + 8차 MVP
================================================

1~6차 MVP의 모든 발견을 통합한 실전 시뮬레이션:

  1차: 페어 발굴 (§1 ADF + §2 공적분 + Half-life + TLS)  →  PairsFinder
  2차: K-fold 검증 (§8 Purged + Embargo)                 →  PurgedKFold
  3차: 오버핏 진단 (Sensitivity + Permutation)            →  (선택 적용)
  4차: 정적 베타 권장 (Kalman은 보험용)
  5차: Ensemble은 안정성 보험 (수익은 정적)
  6차: 매월 페어 풀 갱신 필수                              →  RollingPairsManager
  7차: 다중 페어 동시 운용 + 포트폴리오 켈리              →  이 모듈

8차 MVP 추가: 페어 품질 필터링
  - max_active_pairs: 매월 운용 페어 cap (신호 농축)
  - min_historical_sharpe: 과거 Sharpe 음수 페어 차단 (악성 제거)
  - quality_lookback: 과거 성과 평가 윈도우 (일)

7차 발견: 소수의 악성 페어(AMD 계열)가 좋은 페어 10개 수익 다 까먹음
→ 8차 가설: 사전 Sharpe 필터 + 페어 수 cap이면 회복 가능

비유로 풀면:
  - 단일 페어 운용 = 한 종목만 사는 것
  - 다중 페어 운용 = 분산 투자 (10~15개 페어 동시 운용)
  - 포트폴리오 켈리 = 각 페어에 얼마씩 배분할지 자동 결정
  - 품질 필터 = 매월 면접관 자세로 페어 평가, 불합격은 그 달 빼버림
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from pairs_trading_mvp import (
    PairsFinder, SignalGenerator, KellySizer, Backtester,
    Pair, compute_spread,
)
from rolling_pairs import RollingPairsManager


@dataclass
class MonthlyPosition:
    """매월의 운용 상태"""
    date: pd.Timestamp
    active_pairs: list[Pair]
    pair_weights: dict[str, float]  # 페어ID → 자본 비중
    pair_pnl: dict[str, float]       # 페어ID → 이 달 누적 PnL
    portfolio_pnl: float = 0.0


@dataclass
class PortfolioBacktestResult:
    """전체 시뮬레이션 결과"""
    monthly_states: list[MonthlyPosition]
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: dict
    pair_lifetime_pnl: dict[str, float]  # 페어별 총 누적 PnL


class ProductionSystem:
    """
    1~6차 MVP를 통합한 실전 시뮬레이션 시스템.

    매월 1회:
      1. RollingPairsManager로 페어 풀 갱신
      2. 살아남은 + 신규 페어에 대해 spread/Z-score 계산
      3. 포트폴리오 켈리로 페어별 자본 비중 결정
      4. 신호 발생 시 체결 (delta neutral)

    매일:
      5. 각 페어의 일별 수익률 계산
      6. 종합 자본 곡선 갱신
    """

    def __init__(self,
                 finder: PairsFinder,
                 lookback_days: int = 365,
                 refresh_every_days: int = 30,
                 sig_entry: float = 2.0,
                 sig_exit: float = 0.5,
                 sig_stop: float = 3.5,
                 initial_capital: float = 100_000.0,
                 capital_per_pair_cap: float = 0.10,  # 단일 페어 최대 10%
                 fee_rate: float = 0.0004,
                 slippage: float = 0.0005,
                 portfolio_kelly_fraction: float = 0.25,
                 use_history_for_kelly: bool = True,
                 # ===== 8차 MVP 튜닝 파라미터 =====
                 max_active_pairs: Optional[int] = None,
                 min_historical_sharpe: Optional[float] = None,
                 quality_lookback: int = 90):
        self.finder = finder
        self.lookback_days = lookback_days
        self.refresh_every_days = refresh_every_days
        self.sig_entry = sig_entry
        self.sig_exit = sig_exit
        self.sig_stop = sig_stop
        self.initial_capital = initial_capital
        self.capital_per_pair_cap = capital_per_pair_cap
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.portfolio_kelly_fraction = portfolio_kelly_fraction
        self.use_history_for_kelly = use_history_for_kelly
        # 8차 튜닝
        self.max_active_pairs = max_active_pairs
        self.min_historical_sharpe = min_historical_sharpe
        self.quality_lookback = quality_lookback
        # 진단용: 필터로 컷된 페어 추적
        self.filter_log: list[dict] = []

    def _filter_and_cap_pairs(self,
                              active_pair_ids: list[str],
                              pair_results: dict,
                              current_date: pd.Timestamp,
                              price_df: pd.DataFrame) -> tuple[list[str], dict]:
        """
        8차 MVP 핵심: 페어 품질 필터링 + 운용 수 cap.

        과거 quality_lookback 일간의 일별 수익률로 Sharpe 계산 → 두 가지 필터 적용:
          (1) min_historical_sharpe: 음수/저조한 페어 제거
          (2) max_active_pairs: 상위 N개만 남김

        반환: (필터 통과한 페어 ID, 각 페어의 historical Sharpe dict)

        엣지케이스:
          - 초기 기간이라 lookback 데이터가 20일 미만이면 필터 적용 안 함 (charitable)
          - 두 파라미터 모두 None이면 원본 그대로 반환
        """
        # No filtering requested
        if (self.min_historical_sharpe is None
                and self.max_active_pairs is None):
            return active_pair_ids, {}

        try:
            loc = price_df.index.get_loc(current_date)
        except KeyError:
            return active_pair_ids, {}

        lookback_start = max(0, loc - self.quality_lookback)
        lookback_end = loc

        # 초기 기간 — 충분한 데이터 없으면 필터 보류
        if lookback_end - lookback_start < 20:
            return active_pair_ids, {}

        # 각 페어의 과거 Sharpe 계산
        pair_sharpes: dict[str, float] = {}
        for pid in active_pair_ids:
            if pid not in pair_results:
                continue
            daily_ret = pair_results[pid]['daily_return'].iloc[
                lookback_start:lookback_end].dropna()
            if len(daily_ret) < 10 or daily_ret.std() == 0:
                # 정보 부족 → 0으로 둠 (min_sharpe=0이면 컷, 음수면 통과)
                pair_sharpes[pid] = 0.0
                continue
            sh = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            pair_sharpes[pid] = sh

        # (1) min_historical_sharpe 필터
        if self.min_historical_sharpe is not None:
            filtered = [pid for pid, sh in pair_sharpes.items()
                        if sh >= self.min_historical_sharpe]
        else:
            filtered = list(pair_sharpes.keys())

        # (2) max_active_pairs cap (Sharpe 상위 N)
        if (self.max_active_pairs is not None
                and len(filtered) > self.max_active_pairs):
            filtered = sorted(filtered,
                              key=lambda p: pair_sharpes[p],
                              reverse=True)[:self.max_active_pairs]

        # 진단 로그
        self.filter_log.append({
            'date': current_date,
            'candidates': len(active_pair_ids),
            'after_sharpe_filter': len([p for p in pair_sharpes
                                        if self.min_historical_sharpe is None
                                        or pair_sharpes[p] >= self.min_historical_sharpe]),
            'final': len(filtered),
            'pair_sharpes': dict(pair_sharpes),
            'selected': list(filtered),
        })

        return filtered, pair_sharpes

    def run(self, price_df: pd.DataFrame,
            verbose: bool = False) -> PortfolioBacktestResult:
        """전체 시뮬레이션 실행"""

        # 1. 롤링 페어 풀 갱신
        manager = RollingPairsManager(
            finder=self.finder,
            lookback_days=self.lookback_days,
            refresh_every_days=self.refresh_every_days,
        )
        snapshots = manager.run(price_df, verbose=False)

        # 2. 각 페어의 전체 백테스트 결과 미리 계산
        # (snapshot에서 등장한 모든 페어에 대해 한 번씩)
        all_pair_ids = set()
        pair_objects = {}
        for s in snapshots:
            for p in s.pairs:
                pid = f'{p.y}~{p.x}'
                if pid not in pair_objects:
                    pair_objects[pid] = p
                    all_pair_ids.add(pid)

        if verbose:
            print(f'  전체 페어 후보: {len(all_pair_ids)}개')

        pair_results = {}
        for pid, p in pair_objects.items():
            try:
                sig_window = max(20, min(60, int(p.half_life * 2)))
                spread = compute_spread(price_df[p.y], price_df[p.x], p.beta)
                sg = SignalGenerator(window=sig_window,
                                     entry=self.sig_entry,
                                     exit_thr=self.sig_exit,
                                     stop=self.sig_stop)
                z, pos, fc = sg.generate(spread)
                bt = Backtester(initial_capital=100_000,
                                capital_fraction=1.0,
                                fee_rate=self.fee_rate,
                                slippage=self.slippage)
                res = bt.run(price_df[p.y], price_df[p.x],
                            p.beta, pos, fc)
                # 일별 수익률
                daily_ret = res.equity.pct_change().fillna(0)
                pair_results[pid] = {
                    'pair': p,
                    'daily_return': daily_ret,
                    'trades': res.trades,
                    'metrics': res.metrics,
                }
            except Exception as e:
                if verbose:
                    print(f'  Skip {pid}: {e}')

        # 3. 매월 자본 배분 + 일별 시뮬레이션
        monthly_states = []
        portfolio_equity = pd.Series(self.initial_capital,
                                      index=price_df.index, dtype=float)
        portfolio_ret = pd.Series(0.0, index=price_df.index, dtype=float)
        pair_lifetime_pnl = {pid: 0.0 for pid in pair_results}

        for snap_idx, snap in enumerate(snapshots):
            current_date = snap.date
            active_pair_ids = [f'{p.y}~{p.x}' for p in snap.pairs
                              if f'{p.y}~{p.x}' in pair_results]

            # ===== 8차 MVP: 품질 필터 + cap =====
            if (self.max_active_pairs is not None
                    or self.min_historical_sharpe is not None):
                active_pair_ids, _sharpes = self._filter_and_cap_pairs(
                    active_pair_ids, pair_results, current_date, price_df)

            if not active_pair_ids:
                # 이 달엔 운용 페어 없음 → 현금 보유
                state = MonthlyPosition(
                    date=current_date,
                    active_pairs=[],
                    pair_weights={},
                    pair_pnl={},
                )
                monthly_states.append(state)
                continue

            # 4. 포트폴리오 켈리로 자본 비중 결정
            # 각 페어의 과거 수익률 통계로 mu, cov 추정
            if self.use_history_for_kelly and snap_idx > 0:
                lookback_start = max(
                    0, price_df.index.get_loc(current_date) - 90)
                lookback_end = price_df.index.get_loc(current_date)
                hist_returns = pd.DataFrame({
                    pid: pair_results[pid]['daily_return']
                          .iloc[lookback_start:lookback_end]
                    for pid in active_pair_ids
                })
                mu_vec = hist_returns.mean().values * 252
                cov_mat = hist_returns.cov().values * 252
                weights = KellySizer.portfolio(
                    mu_vec, cov_mat,
                    fraction=self.portfolio_kelly_fraction,
                    cap=self.capital_per_pair_cap,
                )
                weights = np.clip(weights, 0, self.capital_per_pair_cap)
                pair_weights = dict(zip(active_pair_ids, weights))
            else:
                # 처음엔 균등 배분
                n = len(active_pair_ids)
                equal_w = min(1.0 / n, self.capital_per_pair_cap)
                pair_weights = {pid: equal_w for pid in active_pair_ids}

            # 5. 이번 달 일별 수익률 합산
            month_start = current_date
            if snap_idx + 1 < len(snapshots):
                month_end = snapshots[snap_idx + 1].date
            else:
                month_end = price_df.index[-1]

            month_dates = price_df.loc[month_start:month_end].index

            for date in month_dates:
                daily_port_ret = 0.0
                for pid, w in pair_weights.items():
                    if date in pair_results[pid]['daily_return'].index:
                        r = pair_results[pid]['daily_return'].loc[date]
                        daily_port_ret += w * r
                        pair_lifetime_pnl[pid] += w * r * self.initial_capital
                portfolio_ret.loc[date] = daily_port_ret

            # 페어별 이 달 PnL
            pair_pnl = {}
            for pid, w in pair_weights.items():
                month_ret = pair_results[pid]['daily_return'].loc[
                    month_start:month_end].sum()
                pair_pnl[pid] = w * month_ret * self.initial_capital

            # 필터 통과한 페어 객체만 monthly state에 기록
            active_pair_objects = [p for p in snap.pairs
                                   if f'{p.y}~{p.x}' in active_pair_ids]
            state = MonthlyPosition(
                date=current_date,
                active_pairs=active_pair_objects,
                pair_weights=pair_weights,
                pair_pnl=pair_pnl,
                portfolio_pnl=sum(pair_pnl.values()),
            )
            monthly_states.append(state)

            if verbose:
                top_3 = sorted(pair_weights.items(),
                              key=lambda x: x[1], reverse=True)[:3]
                top_3_str = ', '.join([f'{p}({w:.1%})' for p, w in top_3])
                print(f'  {current_date.date()}: {len(active_pair_ids)} 페어, '
                      f'Top3 [{top_3_str}]')

        # 6. 자본 곡선 계산
        portfolio_equity = self.initial_capital * (1 + portfolio_ret).cumprod()

        # 7. 성능 지표
        metrics = self._compute_metrics(portfolio_equity, portfolio_ret)

        return PortfolioBacktestResult(
            monthly_states=monthly_states,
            equity_curve=portfolio_equity,
            daily_returns=portfolio_ret,
            metrics=metrics,
            pair_lifetime_pnl=pair_lifetime_pnl,
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
        sharpe = (float(ret.mean() / ret.std() * np.sqrt(252))
                  if ret.std() > 0 else 0)
        cummax = equity.cummax()
        mdd = float((equity / cummax - 1).min())
        calmar = cagr / abs(mdd) if mdd < 0 else np.inf
        # 양수 일수 비율
        positive_days = float((ret > 0).sum()) / max(len(ret), 1)
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'calmar': calmar,
            'positive_days_pct': positive_days,
        }


if __name__ == "__main__":
    print("production_system module loaded.")
    print("  ProductionSystem, MonthlyPosition, PortfolioBacktestResult")
