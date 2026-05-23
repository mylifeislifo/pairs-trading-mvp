"""
Pairs Trading MVP — 명세서 v2 §1~4 구현
=========================================

모듈 구성:
  §1 PairsFinder.is_nonstationary()      - ADF 검정
  §2 PairsFinder.screen_pairs()           - 공적분 + Half-life + TLS
  §3 SignalGenerator.generate()           - Z-score (position + force_close)
  §4 KellySizer.compute()                 - 단일/포트폴리오 켈리
     Backtester.run()                     - 일봉 페어 백테스트
     Metrics                              - Sharpe, MDD, Calmar 등
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
from statsmodels.tsa.stattools import adfuller, coint
from scipy.odr import ODR, Model, RealData
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# §1 + §2  PairsFinder
# =============================================================================

@dataclass
class Pair:
    """공적분 페어 메타데이터"""
    y: str          # 종속 자산 티커
    x: str          # 독립 자산 티커
    beta: float     # 헤지비율 (TLS)
    pvalue: float   # 공적분 p-value
    half_life: float  # 평균회귀 반감기 (일)

    def __repr__(self):
        return (f"Pair({self.y}~{self.x}: β={self.beta:.3f}, "
                f"p={self.pvalue:.4f}, HL={self.half_life:.1f}d)")


class PairsFinder:
    """ADF + 공적분 + Half-life + TLS 헤지비율 통합"""

    def __init__(self,
                 alpha_adf: float = 0.05,
                 pvalue_coint: float = 0.05,
                 max_halflife: float = 30.0,
                 min_halflife: float = 1.0):
        self.alpha_adf = alpha_adf
        self.pvalue_coint = pvalue_coint
        self.max_halflife = max_halflife
        self.min_halflife = min_halflife

    # ---------------- §1 ----------------
    @staticmethod
    def is_nonstationary(series: pd.Series, alpha: float = 0.05) -> bool:
        """ADF 단위근 검정. True면 I(1) 후보."""
        s = np.log(series.dropna())
        if len(s) < 50:
            return False
        try:
            _, pvalue, *_ = adfuller(s, autolag='AIC')
        except Exception:
            return False
        return pvalue > alpha

    # ---------------- §2-C ----------------
    @staticmethod
    def half_life(spread: pd.Series) -> Optional[float]:
        """평균회귀 반감기. lam>=0이면 None."""
        s = spread.dropna()
        s_lag = s.shift(1).dropna()
        delta = s.diff().dropna()
        idx = s_lag.index.intersection(delta.index)
        if len(idx) < 30:
            return None
        lam = np.polyfit(s_lag.loc[idx].values, delta.loc[idx].values, 1)[0]
        if lam >= 0:
            return None
        return -np.log(2) / lam

    # ---------------- §2-D ----------------
    @staticmethod
    def tls_beta(x: np.ndarray, y: np.ndarray) -> float:
        """Total Least Squares 헤지비율 (양 변수 대칭)"""
        def linear(p, xx):
            return p[0] * xx + p[1]
        try:
            data = RealData(x, y)
            odr = ODR(data, Model(linear), beta0=[1.0, 0.0])
            result = odr.run()
            return float(result.beta[0])
        except Exception:
            # 폴백: OLS
            return float(np.polyfit(x, y, 1)[0])

    # ---------------- 통합 ----------------
    def screen_pairs(self, price_df: pd.DataFrame) -> list[Pair]:
        """모든 쌍을 검정해 적격 페어만 반환"""
        log_prices = np.log(price_df.dropna())
        pairs = []
        tickers = list(log_prices.columns)

        # 사전 ADF 캐시
        ns_cache = {t: self.is_nonstationary(log_prices[t], self.alpha_adf)
                    for t in tickers}

        for i, j in combinations(tickers, 2):
            if not (ns_cache[i] and ns_cache[j]):
                continue

            y, x = log_prices[i], log_prices[j]
            try:
                _, pvalue, _ = coint(y, x)
            except Exception:
                continue
            if pvalue >= self.pvalue_coint:
                continue

            beta = self.tls_beta(x.values, y.values)
            spread = y - beta * x
            hl = self.half_life(spread)
            if hl is None or not (self.min_halflife <= hl <= self.max_halflife):
                continue

            pairs.append(Pair(y=i, x=j, beta=beta, pvalue=pvalue, half_life=hl))

        return sorted(pairs, key=lambda p: p.pvalue)


# =============================================================================
# §3  SignalGenerator
# =============================================================================

class SignalGenerator:
    """Z-score 기반 진입/청산 신호. position + force_close 분리."""

    def __init__(self,
                 window: int = 30,
                 entry: float = 2.0,
                 exit_thr: float = 0.0,
                 stop: float = 3.5):
        self.window = window
        self.entry = entry
        self.exit_thr = exit_thr
        self.stop = stop

    def generate(self, spread: pd.Series):
        """
        반환:
          z: Z-score
          position: pd.Series (-1, 0, +1) — 포지션 상태 (시점별 목표)
          force_close: pd.Series bool — True면 강제 청산 + 신규진입 금지
        """
        mu = spread.rolling(self.window).mean().shift(1)
        sigma = spread.rolling(self.window).std().shift(1).clip(lower=1e-8)
        z = (spread - mu) / sigma

        # 1차 신호: 임계치 기반
        raw = pd.Series(np.nan, index=z.index)
        raw[z > self.entry] = -1
        raw[z < -self.entry] = 1
        raw[z.abs() < self.exit_thr] = 0

        # 상태 유지 (entry까지는 0으로 두지 말고 forward fill)
        position = raw.ffill().fillna(0).astype(int)

        # 강제 청산 플래그
        force_close = z.abs() > self.stop
        # 강제 청산되면 position도 0으로
        position = position.where(~force_close, 0)

        return z, position, force_close


# =============================================================================
# §4  KellySizer
# =============================================================================

class KellySizer:
    """단일 거래 + 포트폴리오 켈리"""

    @staticmethod
    def from_trades(returns: pd.Series,
                    fraction: float = 0.25,
                    cap: float = 0.20) -> float:
        """과거 거래 수익률 시계열에서 켈리 계산.
        승률/손익비를 OOS 데이터에서 추출해야 함."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        p = len(wins) / len(returns)
        b = wins.mean() / abs(losses.mean())
        if b <= 0 or p <= 0:
            return 0.0
        f_full = (b * p - (1 - p)) / b
        return float(min(max(0.0, fraction * f_full), cap))

    @staticmethod
    def continuous(mu: float, sigma_sq: float,
                   rf: float = 0.0,
                   fraction: float = 0.5,
                   cap: float = 0.20) -> float:
        if sigma_sq <= 0:
            return 0.0
        f = (mu - rf) / sigma_sq
        return float(min(max(0.0, fraction * f), cap))

    @staticmethod
    def portfolio(mu_vec: np.ndarray,
                  cov_matrix: np.ndarray,
                  rf: float = 0.0,
                  fraction: float = 0.25,
                  cap: float = 0.20) -> np.ndarray:
        """다중 전략 동시 운용. Σ^-1(μ - rf·1)"""
        excess = mu_vec - rf
        cov_reg = cov_matrix + np.eye(len(mu_vec)) * 1e-8
        f_star = np.linalg.solve(cov_reg, excess)
        return np.clip(fraction * f_star, 0, cap)


# =============================================================================
# Backtester  (일봉 페어 트레이딩)
# =============================================================================

@dataclass
class BacktestResult:
    equity: pd.Series          # 자본 곡선
    daily_pnl: pd.Series       # 일별 손익
    position: pd.Series        # 일별 포지션
    trades: pd.DataFrame       # 거래 로그
    metrics: dict              # 성능 지표


class Backtester:
    """
    페어 트레이딩 백테스트 엔진.

    가정:
      - 일봉 종가 기준 체결
      - 양쪽 다리에 동일 명목가치 (delta neutral)
      - 거래비용: 진입/청산 시점에 양쪽 다리 모두 차감
      - 슬리피지: 명목가치 비율
      - look-ahead 차단: position[t-1] → return[t] 곱
    """

    def __init__(self,
                 initial_capital: float = 100_000.0,
                 capital_fraction: float = 0.10,    # 켈리 출력
                 fee_rate: float = 0.0004,           # 한쪽 다리 0.04% (왕복 0.08%)
                 slippage: float = 0.0005):          # 한쪽 다리 0.05%
        self.initial_capital = initial_capital
        self.capital_fraction = capital_fraction
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(self,
            price_y: pd.Series,
            price_x: pd.Series,
            beta: float,
            position: pd.Series,
            force_close: pd.Series) -> BacktestResult:

        df = pd.DataFrame({
            'py': price_y, 'px': price_x,
            'pos': position.astype(int),
            'force': force_close.astype(bool),
        }).dropna()

        # 로그수익률
        df['ry'] = np.log(df['py']).diff()
        df['rx'] = np.log(df['px']).diff()

        # 포지션 변경 시점 (거래 발생)
        df['pos_prev'] = df['pos'].shift(1).fillna(0).astype(int)
        df['trade'] = df['pos'] != df['pos_prev']

        # 거래 비용: 한 번의 포지션 변경 = 양 다리 체결
        # |Δpos| 만큼 양 다리 거래 발생. 비용 = fee + slippage, 양 다리니까 ×2
        cost_per_change = (self.fee_rate + self.slippage) * 2
        df['cost'] = df['trade'].astype(float) * cost_per_change

        # 페어 수익률 (delta neutral 가정, 양 다리 동일 명목가치):
        #   strat_ret = pos_{t-1} * (ry_t - rx_t)
        # beta는 헤지비율로 들어가지만, 실무에선 명목가치 동등 비율이 더 흔함.
        # 더 정확히는 spread return = ry - β*rx 이지만, 자본 배분 시
        # x 다리에 β를 곱해 명목가치 늘리면 자본 사용량 비대칭이 됨.
        # 여기서는 표준적인 dollar-neutral 가정 (양쪽 동일 명목).
        df['strat_ret'] = df['pos_prev'] * (df['ry'] - df['rx']) - df['cost']

        # 자본 곡선
        # capital_fraction만큼만 페어에 노출. 나머지는 현금.
        # 첫 행 NaN이 cumprod 전체를 오염시키므로 fillna(0) 필수.
        df['daily_pnl_pct'] = (df['strat_ret'] * self.capital_fraction).fillna(0)
        df['equity'] = self.initial_capital * (1 + df['daily_pnl_pct']).cumprod()

        # 거래 로그
        trades = self._extract_trades(df)

        # 성능 지표
        metrics = self._compute_metrics(df['equity'], df['daily_pnl_pct'], trades)

        return BacktestResult(
            equity=df['equity'],
            daily_pnl=df['daily_pnl_pct'] * self.initial_capital,
            position=df['pos'],
            trades=trades,
            metrics=metrics,
        )

    @staticmethod
    def _extract_trades(df: pd.DataFrame) -> pd.DataFrame:
        """거래 시작-종료 쌍 추출"""
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
                    'pnl_pct': (row['equity'] / entry_equity - 1) if entry_equity else 0,
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
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1 if n_years > 0 else 0

        ret = daily_ret.dropna()
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

        cummax = equity.cummax()
        drawdown = (equity / cummax - 1)
        mdd = drawdown.min()

        calmar = cagr / abs(mdd) if mdd < 0 else np.inf

        win_rate = (trades['pnl_pct'] > 0).mean() if len(trades) > 0 else 0
        n_trades = len(trades)
        avg_duration = trades['duration'].mean() if len(trades) > 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'calmar': calmar,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_trade_duration': avg_duration,
        }


# =============================================================================
# 편의 함수
# =============================================================================

def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7):
    """시간 순 분할 (look-ahead 차단)"""
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]


def compute_spread(price_y: pd.Series, price_x: pd.Series, beta: float) -> pd.Series:
    """log spread = log(y) - β·log(x)"""
    return np.log(price_y) - beta * np.log(price_x)


if __name__ == "__main__":
    print("pairs_trading_mvp module loaded.")
    print("  PairsFinder, SignalGenerator, KellySizer, Backtester")
