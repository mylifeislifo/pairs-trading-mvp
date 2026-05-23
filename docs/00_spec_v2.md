# 제인스트리트 차용 수학 알고리즘 — 구현 명세서 v2

> **v2 변경 이력 (2026.05.23)**
> v1의 검증 과정에서 발견된 **명확한 오류 4건 + 부정확/모호 4건**을 전부 수정했다.
> 1. §10 펀딩비 손익 부호 + 거래 비용 임계치 (전략 손실 직결)
> 2. §9 VWAP 세션 리셋 + Max Pain 만기 분리
> 3. §2 Half-life 부호·정의 명확화
> 4. §3 Z-score 손절 플래그 boolean화
> 5. §2 OLS 헤지비율 → TLS / Kalman 대안 추가
> 6. §8 Purge / Embargo 분리 (López de Prado 정통 구현)
> 7. §4 다중 포지션 통합 켈리 수식 추가
> 8. 모든 의사코드 실행 가능성 재검토

---

## 0. 알고리즘 의존성 맵

```
[데이터 입력 계층]
      ↓
[1] 정상성 검정 (ADF) ──→ [2] 공적분 검정 (EG/Johansen + Half-life)
                              ↓
                         [3] Z-score 진입/청산 (position + force_close)
                              ↓
                         [4] 켈리 기준 사이징 (단일/포트폴리오)
                              ↓
                         [최종 주문 + 슬리피지 사전 검증]
                              ↑
[5] GBM Drift 추정 ──→ [6] MDN 손실 ──→ [예측 신호]
[7] 칼만 필터 (마이크로 가격 / 동적 베타)
[8] Purged + Embargo Split ──→ (모든 ML 모듈의 검증 전처리)
[9] VWAP/Max Pain ──→ (만기일 평균회귀)
[10] 펀딩비 (베이시스 + 거래비용 사전 검증) ──→ (델타 뉴트럴)
```

**핵심 원칙**:
- 검증 계층(§8)은 모든 ML 모듈의 전처리 단계로 반드시 선행
- §10은 진입 신호만으로 부족, **누적 펀딩 - 왕복 비용** 손익분기 체크 필수
- §2는 단순 OLS 헤지비율 대신 **TLS 또는 Kalman 동적 베타** 권장

---

## 1. ADF 단위근 검정 (Augmented Dickey-Fuller)

**역할**: 페어 트레이딩 진입 전, 두 자산 가격이 각각 비정상성(I(1))인지 확인.

### 입력
- `price_series`: 1차원 시계열 (최소 250개 관측치 권장)

### 핵심 수식
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

- $H_0: \gamma = 0$ (단위근 존재, 비정상)
- $H_1: \gamma < 0$ (정상)

### 파라미터
| 파라미터 | 권장값 | 의미 |
|---|---|---|
| `maxlag` | $\lfloor 12(T/100)^{1/4} \rfloor$ (Schwert 1989) | AIC/BIC로 자동 선택 |
| `regression` | `'c'` 또는 `'ct'` | 상수항/추세항 포함 여부 |
| 유의수준 | 0.05 | p-value > 0.05면 비정상 |

### 라이브러리
```python
from statsmodels.tsa.stattools import adfuller
```

### 의사코드
```python
def is_nonstationary(series, alpha=0.05):
    """반환 True면 I(1) 후보 (공적분 검정 대상)"""
    series = np.log(series.dropna())  # 반드시 로그가격
    adf_stat, pvalue, *_ = adfuller(series, autolag='AIC')
    return pvalue > alpha
```

### 실전 함정
- 로그가격(`log P`) 필수. 원가격은 분산 비균질.
- 표본 < 100이면 검정력 부족.
- 구조적 단절(코로나, 911 등) 포함된 구간은 검정 왜곡.

---

## 2. 공적분 검정 + Half-life + 헤지비율 추정

**역할**: 두 비정상 시계열의 선형 결합이 정상인지 확인 + 평균회귀 속도 측정.

### 2-A. Engle-Granger (두 자산)

#### 핵심 수식
1단계: $Y_t = \alpha + \beta X_t + u_t$ (OLS)
2단계: 잔차 $\hat{u}_t$에 ADF 검정

```python
from statsmodels.tsa.stattools import coint
score, pvalue, crit = coint(y, x)
```

### 2-B. Johansen (다자산)

VECM 기반:
$$\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta Y_{t-i} + \epsilon_t$$

$\Pi$의 랭크 $r$이 공적분 벡터 개수.

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
result = coint_johansen(df, det_order=0, k_ar_diff=1)
```

### 2-C. Half-life (평균회귀 속도) — [v2 수정]

#### 정의 명확화
AR(1) 잔차 모델:
$$\Delta s_t = \lambda \cdot s_{t-1} + \epsilon_t$$

- `lam < 0`이면 평균회귀 (안정)
- `lam >= 0`이면 페어 폐기

$$\text{Half-life} = -\frac{\ln 2}{\lambda}, \quad (\lambda < 0 \text{일 때만 유효})$$

OU 모델 표기와의 관계: $\theta = -\lambda$이므로 Half-life $= \ln 2 / \theta$ (양수).

#### 의사코드
```python
def half_life(spread):
    """평균회귀 반감기. lam >= 0이면 None 반환 (페어 폐기 신호)"""
    spread = spread.dropna()
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    # 인덱스 정렬
    common_idx = spread_lag.index.intersection(delta.index)
    lam = np.polyfit(spread_lag.loc[common_idx], 
                      delta.loc[common_idx], 1)[0]
    if lam >= 0:
        return None  # 평균회귀 없음
    return -np.log(2) / lam
```

### 2-D. 헤지비율 추정 — [v2 수정: OLS → TLS]

#### 문제
`np.polyfit(x, y, 1)`로 구한 $\beta_{yx}$와 `polyfit(y, x, 1)`로 구한 $\beta_{xy}$는 일반적으로 다름 ($\beta_{yx} \cdot \beta_{xy} = R^2 \neq 1$). 어느 자산을 종속변수로 둘지 임의적.

#### 해결책 1: Total Least Squares (직교 회귀)
```python
from scipy.odr import ODR, Model, RealData

def tls_beta(x, y):
    """양 변수 대칭적인 헤지비율"""
    def linear(p, x): return p[0] * x + p[1]
    data = RealData(x, y)
    odr = ODR(data, Model(linear), beta0=[1.0, 0.0])
    result = odr.run()
    return result.beta[0]
```

#### 해결책 2: Kalman 동적 베타 (§7 참조) — 시변 헤지비율로 가장 견고

### 통합 페어 스크리너 의사코드
```python
def screen_pairs(price_df, pvalue_thresh=0.05, max_halflife=21):
    """반환: (자산1, 자산2, 헤지비율, p-value, half-life)"""
    pairs = []
    for i, j in combinations(price_df.columns, 2):
        y, x = np.log(price_df[i]), np.log(price_df[j])
        
        # 1차 필터: 양쪽 다 비정상
        if not (is_nonstationary(y) and is_nonstationary(x)):
            continue
        
        # 2차 필터: 공적분
        _, pvalue, _ = coint(y, x)
        if pvalue >= pvalue_thresh:
            continue
        
        # 3차 필터: 평균회귀 속도
        beta = tls_beta(x.values, y.values)
        spread = y - beta * x
        hl = half_life(spread)
        if hl is None or hl > max_halflife:
            continue
        
        pairs.append((i, j, beta, pvalue, hl))
    return sorted(pairs, key=lambda p: p[3])
```

### 실전 함정
- OOS 공적분 붕괴 빈번 → **롤링 윈도우 6개월 재검정 필수**.
- 동일 섹터·비즈니스 모델 쌍이 안정. 무작위 페어는 가짜 공적분(spurious) 위험.
- Half-life > 한 달이면 자본 묶임. `max_halflife=21` 정도 권장.

---

## 3. Z-score 진입/청산 — [v2 수정: position + force_close 분리]

### 핵심 수식
$$Z_t = \frac{S_t - \mu_S^{(w)}}{\sigma_S^{(w)}}$$

$\mu_S^{(w)}, \sigma_S^{(w)}$: 윈도우 $w$ 내의 평균·표준편차 (반드시 `shift(1)` 적용).

### 파라미터
| 파라미터 | 권장 범위 | 의미 |
|---|---|---|
| `window` | 20~60일 (일봉) | Half-life × 2~3 권장 |
| `entry_threshold` | ±2.0 | 진입 트리거 |
| `exit_threshold` | ±0.0~±0.5 | 청산 트리거 |
| `stop_loss` | ±3.5~±4.0 | 공적분 붕괴 손절 |

### 의사코드 — [v2 수정]
```python
def generate_signal(spread, window=30, entry=2.0, exit_thr=0.0, stop=3.5):
    """
    반환:
      z: Z-score 시계열
      position: -1 (스프레드 숏) / 0 (중립) / +1 (스프레드 롱)
      force_close: True면 강제 청산 + 신규 진입 금지
    """
    mu = spread.rolling(window).mean().shift(1)  # look-ahead 차단
    sigma = spread.rolling(window).std().shift(1).clip(lower=1e-6)
    z = (spread - mu) / sigma
    
    position = pd.Series(0, index=z.index)
    position[z > entry] = -1   # 스프레드 매도 (y 숏, x 롱)
    position[z < -entry] = 1   # 스프레드 매수 (y 롱, x 숏)
    position[abs(z) < exit_thr] = 0
    
    force_close = (abs(z) > stop)  # boolean으로 분리
    return z, position, force_close

# 체결 로직 예시
def execute(z, position, force_close):
    if force_close.iloc[-1]:
        close_all_positions()
        return  # 신규 진입 금지
    target_pos = position.iloc[-1]
    rebalance_to(target_pos)
```

### 실전 함정
- `rolling().mean()`은 현재 시점 포함 → 반드시 `shift(1)`.
- σ가 0 근처면 Z 발산 → 분모 클리핑.
- Z가 3~4로 발산하면 평균회귀 가정 깨진 것 → **Half-life × 3 시간 내 회귀 없으면 강제 청산** 룰 권장.
- 윈도우 길이는 §2의 Half-life × 2~3이 합리적.

---

## 4. 켈리 기준 — 단일 + 포트폴리오 [v2 보강]

### 4-A. 단일 거래 (이산형)
$$f^* = \frac{bp - q}{b}$$
- $p$: 승률, $q = 1-p$, $b$: 손익비

### 4-B. 단일 거래 (연속형, 정규근사)
$$f^* = \frac{\mu - r_f}{\sigma^2}$$

### 4-C. 다중 자산 포트폴리오 켈리 — [v2 신규]

상관행렬 고려한 최적 비중:
$$\mathbf{f}^* = \Sigma^{-1}(\boldsymbol{\mu} - r_f \mathbf{1})$$

- $\boldsymbol{\mu}$: 각 전략의 기대수익률 벡터
- $\Sigma$: 전략 간 공분산 행렬
- 단순 합산하면 총 노출 100% 초과 → 반드시 공분산으로 정규화

### 파라미터
| 파라미터 | 권장값 |
|---|---|
| `fraction` | 0.25~0.5 (Fractional Kelly) |
| `cap` | 0.10~0.20 (단일 거래 상한) |
| `lookback` | 100~500 거래 (OOS 통계만 사용) |

### 의사코드
```python
def kelly_sizing(win_rate, win_loss_ratio, fraction=0.25, cap=0.20):
    """단일 거래"""
    p, q, b = win_rate, 1 - win_rate, win_loss_ratio
    if b <= 0 or p <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return min(max(0, fraction * f_full), cap)

def portfolio_kelly(mu_vec, cov_matrix, rf=0.0, 
                     fraction=0.25, cap=0.20):
    """다중 전략 동시 운용"""
    excess = mu_vec - rf
    # 공분산 정칙화 (numerical stability)
    cov_reg = cov_matrix + np.eye(len(mu_vec)) * 1e-8
    f_star = np.linalg.solve(cov_reg, excess)
    return np.clip(fraction * f_star, 0, cap)

def kelly_continuous(mu, sigma_sq, rf=0.0, fraction=0.5, cap=0.20):
    """연속형 단일 자산"""
    if sigma_sq <= 0:
        return 0.0
    f = (mu - rf) / sigma_sq
    return min(max(0, fraction * f), cap)
```

### 실전 함정
- 인샘플 통계로 추정하면 과적합 → **반드시 OOS 통계 사용**.
- 풀 켈리는 일중 드로다운 -50%까지. **Half-Kelly 또는 Quarter-Kelly** 권장.
- 손익비 $b$ 계산 시 단순 평균보다 **CVaR(95%)로 꼬리 보정**.
- 포트폴리오 켈리에서 공분산 추정 자체가 노이즈 → Ledoit-Wolf 축소 추정 권장.

---

## 5. GBM 표류(Drift) 추정

### 핵심 수식

GBM:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

이토 보조정리:
$$r_t = \ln(S_t/S_{t-1}) = (\mu - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} \cdot Z$$

MLE 추정량:
$$\hat{\mu} = \frac{\bar{r}}{\Delta t} + \frac{\hat{\sigma}^2}{2}, \quad \hat{\sigma}^2 = \frac{\text{Var}(r)}{\Delta t}$$

### 파라미터
| 파라미터 | 권장값 |
|---|---|
| `window` | 60~252 거래일 (일봉) |
| `dt` | 1/252 (연환산) |
| 다중 시간지평 | `resp_1, resp_2, resp_3, resp_4` 다양화 |

### 의사코드
```python
def estimate_gbm_params(log_prices, window=60, dt=1/252):
    log_returns = log_prices.diff().dropna()
    mu_bar = log_returns.rolling(window).mean()
    var_r = log_returns.rolling(window).var()
    sigma_sq = var_r / dt
    mu = mu_bar / dt + sigma_sq / 2
    return mu, np.sqrt(sigma_sq)
```

### 실전 함정
- GBM은 팻테일 미포착 → Merton Jump-Diffusion / SVJ 확장 고려.
- σ 추정에 GARCH(1,1) 결합 권장: 
  $$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$
- 단기(<5분) 데이터는 마이크로구조 노이즈로 σ 과대추정 → Two-scale Realized Volatility 보정.

---

## 6. MDN 손실함수 (Mixture Density Network)

### 핵심 수식
$K$개 가우시안 혼합:
$$p(y|x) = \sum_{i=1}^{K} \pi_i(x) \cdot \mathcal{N}(y \mid \mu_i(x), \sigma_i^2(x))$$

손실 (NLL):
$$\mathcal{L} = -\frac{1}{N}\sum_n \log \sum_i \pi_i^{(n)} \mathcal{N}(y^{(n)} \mid \mu_i^{(n)}, \sigma_i^{(n)2})$$

### 출력층
| 출력 | 활성화 | 차원 |
|---|---|---|
| $\pi_i$ | Softmax | $K$ |
| $\mu_i$ | Linear | $K$ |
| $\sigma_i$ | Softplus + 하한 | $K$ |

### 의사코드 (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDNHead(nn.Module):
    def __init__(self, hidden_dim, K=5):
        super().__init__()
        self.K = K
        self.pi = nn.Linear(hidden_dim, K)
        self.mu = nn.Linear(hidden_dim, K)
        self.sigma = nn.Linear(hidden_dim, K)
    
    def forward(self, h):
        pi = F.softmax(self.pi(h), dim=-1)
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + 1e-6  # 하한 필수
        return pi, mu, sigma

def mdn_nll(y, pi, mu, sigma, entropy_weight=0.01):
    """logsumexp 사용한 수치 안정 NLL + 모드붕괴 방지"""
    log_probs = -0.5 * ((y.unsqueeze(-1) - mu) / sigma) ** 2 \
                - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_mix = torch.log(pi + 1e-12) + log_probs
    nll = -torch.logsumexp(log_mix, dim=-1).mean()
    
    # 엔트로피 정규화 (모드 붕괴 방지)
    entropy = -(pi * torch.log(pi + 1e-12)).sum(dim=-1).mean()
    return nll - entropy_weight * entropy
```

### 실전 함정
- 모드 붕괴 (π가 한 컴포넌트로 쏠림) → 엔트로피 정규화 추가.
- σ 하한 클리핑 없으면 NLL 발산.
- K는 3~7 실용 범위.

---

## 7. 칼만 필터 — 동적 헤지비율 / 마이크로 가격

### 핵심 수식

상태/관측 방정식:
$$x_t = F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$z_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

재귀 갱신:
$$\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}, \quad P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (z_t - H \hat{x}_{t|t-1})$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

### 적용: 동적 헤지비율 $\beta_t$ 추정

$y_t = \beta_t x_t + \epsilon_t$에서 $\beta_t$를 잠재 상태로 모델링:
- $F = 1$ (랜덤워크 가정)
- $H = x_t$ (관측 시점 가격)
- $Q$: 베타 변동성 (작게)
- $R$: 가격 노이즈 분산

### 의사코드 (수기 구현)
```python
def kalman_dynamic_beta(y, x, delta=1e-5, R=1e-3):
    """
    delta: 베타 변동성 제어 (작을수록 베타가 천천히 변함)
    R: 관측 노이즈 분산
    """
    n = len(y)
    beta = np.zeros(n)
    P = 1.0
    Q = delta / (1 - delta)
    
    for t in range(n):
        # 예측
        beta_pred = beta[t-1] if t > 0 else 0
        P_pred = P + Q
        
        # 갱신
        H = x[t]
        K = P_pred * H / (H * H * P_pred + R)
        beta[t] = beta_pred + K * (y[t] - H * beta_pred)
        P = (1 - K * H) * P_pred
    
    return beta
```

### 라이브러리 대안 (EM 자동 튜닝)
```python
from pykalman import KalmanFilter
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[[1]],
    initial_state_mean=0,
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.01
)
# EM으로 Q, R 자동 추정
kf = kf.em(y_data, n_iter=10)
state_means, state_covs = kf.filter(y_data)
```

### 실전 함정
- $Q, R$ 튜닝이 핵심 → **EM 알고리즘으로 MLE**.
- 비선형 동역학이면 EKF/UKF.
- 레짐 체인지 시 Q 일시 증대 (adaptive Kalman).

---

## 8. Purged + Embargo Time-Series Split — [v2 수정]

### 정통 정의 (López de Prado)
| 요소 | 의미 |
|---|---|
| **Purge** | 검증셋과 라벨 시간 윈도우가 겹치는 **이전** 훈련 샘플 제거 |
| **Embargo** | 검증셋 **직후** 일정 비율 추가 배제 (검증→훈련 자기상관 차단) |

**핵심**: Purge는 검증셋 **앞쪽만**, Embargo는 검증셋 **뒤쪽만**. v1 의사코드는 양쪽 동일 gap으로 처리해 정통과 다름.

### 파라미터
| 파라미터 | 권장값 |
|---|---|
| `n_splits` | 5~10 |
| `purge_days` | 라벨 lookahead 최대 길이 + α (예: N일 수익률 라벨이면 N+1일) |
| `embargo_pct` | 1~2% (전체 데이터의 비율) |

### 의사코드 — [v2 수정]
```python
def purged_kfold_with_embargo(times, n_splits=5, 
                                purge_days=31, embargo_pct=0.01):
    """
    정통 López de Prado 구현:
    - Purge: 검증셋 시작 시점 이전 purge_days만큼 훈련셋에서 제거
    - Embargo: 검증셋 끝난 직후 embargo 크기만큼 훈련셋에서 제거
    """
    n = len(times)
    fold_size = n // n_splits
    embargo = int(n * embargo_pct)
    
    for k in range(n_splits):
        val_start = k * fold_size
        val_end = val_start + fold_size
        
        train_mask = np.ones(n, dtype=bool)
        # Purge: 검증셋 앞쪽 (검증셋 자체 포함)
        train_mask[max(0, val_start - purge_days):val_end] = False
        # Embargo: 검증셋 직후
        train_mask[val_end:min(n, val_end + embargo)] = False
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.arange(val_start, val_end)
        yield train_idx, val_idx
```

### 라이브러리 대안
```python
from mlfinlab.cross_validation import PurgedKFold
# 정통 구현이지만 유료 라이브러리. 위 수기 구현으로 충분
```

### 실전 함정
- 라벨이 미래 N일 수익률이면 `purge_days >= N+1` 필수.
- 다중 자산이면 시점이 아닌 (자산, 시점) 그룹 단위로 분할.
- 검증 결과가 너무 좋으면 leakage 의심 → embargo 늘려 재검증.

---

## 9. VWAP / Max Pain — [v2 수정: 세션 리셋 + 만기 분리]

### 9-A. VWAP — [v2 수정]

#### 핵심 수식
$$\text{VWAP}_t^{(\text{session})} = \frac{\sum_{i \in \text{session}, i \leq t} P_i V_i}{\sum_{i \in \text{session}, i \leq t} V_i}$$

#### 의사코드 — 세션별 리셋 적용
```python
def vwap_session(df):
    """
    df: DatetimeIndex, columns=['price', 'volume']
    매일 세션 시작 시 누적값 리셋
    """
    df = df.copy()
    df['date'] = df.index.date
    df['pv'] = df['price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['cum_v'] = df.groupby('date')['volume'].cumsum()
    return df['cum_pv'] / df['cum_v']

def vwap_deviation_signal(price, vwap_series, 
                          std_window=20, threshold_sigma=2.0):
    """VWAP 대비 편차의 Z-score 기반 평균회귀 신호"""
    deviation = price - vwap_series
    sigma = price.diff().rolling(std_window).std().shift(1)
    z_dev = deviation / sigma.clip(lower=1e-6)
    
    # 편차가 큰 방향의 반대로 진입
    signal = pd.Series(0, index=price.index)
    signal[z_dev > threshold_sigma] = -1  # 과상승 → 숏
    signal[z_dev < -threshold_sigma] = 1  # 과하락 → 롱
    return signal
```

### 9-B. Max Pain — [v2 수정: 만기 분리]

#### 핵심 수식
$$\text{Pain}(K) = \sum_j OI_j^{\text{call}} \max(K - K_j, 0) + \sum_j OI_j^{\text{put}} \max(K_j - K, 0)$$

#### 의사코드 — 만기별 분리 처리
```python
def max_pain(option_chain, expiry):
    """
    option_chain: DataFrame ['strike', 'call_oi', 'put_oi', 'expiry']
    expiry: 단일 만기일만 처리 (중복 strike 방지)
    """
    chain = option_chain[option_chain['expiry'] == expiry].copy()
    # 같은 strike 중복 제거
    chain = chain.groupby('strike').agg({
        'call_oi': 'sum', 
        'put_oi': 'sum'
    }).reset_index()
    
    strikes = chain['strike'].values
    pain = []
    for K in strikes:
        call_p = ((K - chain['strike']).clip(lower=0) 
                  * chain['call_oi']).sum()
        put_p = ((chain['strike'] - K).clip(lower=0) 
                 * chain['put_oi']).sum()
        pain.append(call_p + put_p)
    return strikes[np.argmin(pain)]

def max_pain_all_expiries(option_chain):
    """모든 만기에 대해 일괄 계산"""
    return {
        exp: max_pain(option_chain, exp)
        for exp in option_chain['expiry'].unique()
    }
```

### 실전 함정
- VWAP은 누적값 → 매일 세션 리셋 필수 (의사코드에 반영됨).
- Max Pain은 사후 통계, 실제 종가와 ±2% 괴리 → **보조지표로만**, 단독 진입 금지.
- 만기 1일 전에는 OI가 급변 → 만기 당일 새벽 데이터로 재계산 권장.

---

## 10. 펀딩비 캐리 (델타 뉴트럴) — [v2 대폭 수정]

### 손익 구조 — [v2 부호 수정]

| 포지션 | 페이오프 |
|---|---|
| Spot Long | $S_T - S_0$ |
| Perp Short | $F_0 - F_T + \sum_t f_t \cdot S_t$ (**+**: 펀딩 양수 시 short가 수취) |
| **합계** | $\approx +\sum_t f_t \cdot S_t$ (베이시스 무시) |

**핵심**: $F \approx S$ 수렴 가정하에 가격 변동은 상쇄, **펀딩비 누적만 남음**.

### 연환산
$$r_{annual} = f_t \cdot \frac{365 \times 24}{\tau_{hours}}$$

8시간 주기면 연 $f_t \times 1095$.

### 진입 조건 — [v2 신규: 거래 비용 손익분기 체크]

#### 왕복 비용 산정
- 4회 체결: spot 진입, perp 진입, spot 청산, perp 청산
- 바이낸스 taker 0.04% × 4 = **0.16%** (수수료만)
- 슬리피지 0.05~0.10% (자산별)
- 베이시스 변동 잠재 손실 0.05~0.15%
- **실효 왕복 비용 0.3~0.5%**

#### 손익분기 조건
한 번 펀딩 사이클(8시간) 수취액: $f_t$ (예: 0.01% = 0.0001)
연 20% APR이면 회당 펀딩 $\approx 0.018\%$
**왕복 비용 0.3% 회수에 필요한 펀딩 횟수**: $0.003 / 0.00018 \approx 17$회 = **약 6일**

### 의사코드 — [v2 신규: 비용 체크 통합]
```python
def funding_arb_signal(current_funding_rate, 
                       expected_holding_periods=21,
                       round_trip_cost=0.003,
                       safety_margin=1.5):
    """
    매 펀딩 시점마다 호출. 진입 여부 + 청산 여부 모두 판단.
    
    파라미터:
      current_funding_rate: 다음 펀딩 예상 비율 (0.0001 = 0.01%)
      expected_holding_periods: 보유 예상 펀딩 횟수 (8시간 단위)
      round_trip_cost: 왕복 거래 비용 비율 (0.003 = 0.3%)
      safety_margin: 손익분기 대비 안전 배율
    
    반환:
      action: 'long_spot_short_perp' / 'short_spot_long_perp' / 'hold' / None
      expected_pnl: 예상 누적 펀딩 수익
    """
    expected_pnl = abs(current_funding_rate) * expected_holding_periods
    breakeven = round_trip_cost * safety_margin
    
    if expected_pnl < breakeven:
        return None, expected_pnl  # 진입 금지
    
    if current_funding_rate > 0:
        return 'long_spot_short_perp', expected_pnl
    else:
        return 'short_spot_long_perp', expected_pnl


def funding_arb_exit_check(current_funding_rate, 
                            cumulative_funding_pnl,
                            entry_cost_paid,
                            min_remaining_apr=0.10):
    """
    이미 진입한 포지션의 청산 여부 판단.
    펀딩비 부호 전환 또는 임계치 하향 돌파 시 청산.
    """
    annualized = current_funding_rate * (365 * 24 / 8)
    
    # 이미 손익분기 넘었으면 신호 약해도 일부 보유 가능
    net_pnl = cumulative_funding_pnl - entry_cost_paid
    
    # 부호 전환 → 즉시 청산
    if current_funding_rate * cumulative_funding_pnl < 0:
        return 'close_now', net_pnl
    
    # APR 임계치 하향
    if abs(annualized) < min_remaining_apr:
        return 'close_now', net_pnl
    
    return 'hold', net_pnl


def position_sizes(capital, asset_price, leverage=1.0, 
                    safety_buffer=2.0):
    """
    spot/perp 양쪽 동등 명목가치.
    유지증거금의 safety_buffer배만큼 여유 자본 확보.
    """
    notional_per_side = capital / (2 * safety_buffer)
    spot_qty = notional_per_side / asset_price
    perp_qty = notional_per_side * leverage / asset_price
    return spot_qty, perp_qty
```

### 실전 함정 — [v2 대폭 보강]
- **거래 비용 미계산 시 거의 무조건 손실**. 단발 진입 절대 금지.
- 베이시스 리스크: 진입·청산 시점의 현·선물 가격 격차. 슬리피지 사전 계산.
- 거래소 청산 위험: 레버리지 ≥ 2배면 변동성 폭발 시 강제청산 → **레버리지 1배 + 유지증거금 200%↑**.
- 펀딩비 음수 전환: 매 펀딩 시점마다 부호 모니터링, 임계치 하향 돌파 시 즉시 청산.
- 거래소 단일 실패점 (해킹·출금정지) → 자본 50% 이상 단일 거래소 집중 금지.
- 자금조달비율이 비정상적으로 높을 때(연 100%+) → **시장이 무언가를 가격에 반영 중**. 무지성 진입 위험.

---

## 부록 A. 모듈 통합 체크리스트

봇 개발 순서:

```
1. 데이터 수집 (OHLCV, 옵션체인, 펀딩비)
   ↓
2. 정상성 검정 [§1] → 페어 후보 필터
   ↓
3. 공적분 검정 + Half-life [§2] → 적격 페어 + TLS 헤지비율
   ↓
4. Z-score 시그널 [§3] (position + force_close 분리)
   ↓ (선택) 칼만 [§7]로 베타 동적화
5. 백테스트 검증 [§8] (Purge + Embargo 분리)
   ↓
6. 켈리 사이징 [§4] (OOS 통계, 다중 페어면 portfolio_kelly)
   ↓
7. 실시간 체결 + 리스크 모니터링
   - 손익분기 미달 거래 차단 (§10 패턴 일반화)
   - force_close 우선 처리
```

## 부록 B. 라이브러리 스택

| 카테고리 | 라이브러리 |
|---|---|
| 통계 검정 | `statsmodels`, `arch` (GARCH) |
| TLS 회귀 | `scipy.odr` |
| ML | `pytorch`, `scikit-learn` |
| 금융 검증 | `mlfinlab` (PurgedKFold, 유료), `vectorbt` |
| 칼만 | `pykalman`, `filterpy` |
| 거래소 API | `ccxt` (통합), `python-binance`, `pyupbit` |
| 데이터 | `yfinance`, `pandas-datareader`, `pykrx` (KR) |

## 부록 C. 절대 모방 금지 영역

| 영역 | 사유 |
|---|---|
| 만기일 호가 폭격 (LTP 누르기/띄우기) | 시세조종, 형사처벌 |
| ETF 발행시장 직접 개입 | AP 라이선스 필요 |
| 13F 종목 단순 추종 | 임시 재고이므로 무의미 |
| Front-running 패시브 펀드 | 규제 회색지대 |

## 부록 D. v1 → v2 수정 요약

| § | v1 문제 | v2 해결 |
|---|---|---|
| 2 | Half-life 부호 모호, OLS 비대칭 | $\lambda < 0$ 조건 + TLS |
| 3 | `signal = 999` 매직넘버 | position + force_close 분리 |
| 4 | 다중 포지션 통합 켈리 누락 | `portfolio_kelly` 추가 |
| 6 | 모드붕괴 방지 미흡 | 엔트로피 정규화 추가 |
| 8 | Purge/Embargo 미분리 | 정통 분리 구현 |
| 9 | VWAP 세션 리셋 누락, Max Pain 만기 미분리 | 둘 다 수정 |
| 10 | 펀딩 부호 오류, 거래비용 미체크 | 부호 + 손익분기 통합 |

---

**문서 끝**
