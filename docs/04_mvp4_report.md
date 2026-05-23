# 4차 MVP — Kalman 동적 베타 보고서

> 1차 MVP의 GDX~KO 실패 원인 — spread 평균값 시프트(-5.38→-4.8) — 을 §7 Kalman 동적 베타로 해결 가능한지 직접 검증.

## 1. 검증 가설

> **"Kalman 동적 베타가 GDX~KO의 fold 3 (regime change 구간) Sharpe -0.29 손실을 줄여준다."**

가설의 의의:
- 만약 지지 → Kalman을 표준 도구로 채택
- 만약 기각 → Kalman만으로는 부족, 롤링 재발굴이 정답
- **결과에 따라 다음 단계가 명확해지는 결정적 실험**

## 2. 구현 — kalman_pairs.py

명세서 v2 §7 의사코드를 그대로 코드화:

```python
def kalman_dynamic_beta(y_log, x_log, delta=1e-5, R=1e-3, warmup=60):
    """
    상태 방정식: β_t = β_{t-1} + w_t
    관측 방정식: y_log_t = β_t · x_log_t + v_t
    """
    # Warmup: 첫 60일은 OLS β로 고정 (transient 회피)
    beta_init = np.polyfit(x_log[:warmup], y_log[:warmup], 1)[0]
    beta[:warmup] = beta_init
    P = 1.0
    Q = delta / (1 - delta)
    
    for t in range(warmup, n):
        beta_pred = beta[t-1]
        P_pred = P + Q
        H = x_log[t]
        K = P_pred * H / (H * H * P_pred + R)
        beta[t] = beta_pred + K * (y_log[t] - H * beta_pred)
        P = (1.0 - K * H) * P_pred
    
    return beta
```

추가 모듈:
- `compute_dynamic_spread()`: β_t 시계열로 시변 spread 계산
- `DynamicBetaBacktester`: spread return 계산 시 β_{t-1}을 곱해 look-ahead 차단
- `kfold_evaluate_pair_dynamic()`: 동적 베타로 K-fold 평가 (정적 버전과 시그니처 호환)

## 3. 핵심 가설 검증 결과 — ✓ 강력히 지지됨

| Fold | 정적 Sharpe | 동적 Sharpe | Δ |
|---|---|---|---|
| 0 | +1.46 | +1.49 | +0.03 (변화 없음) |
| 1 | **+4.94** | **-1.04** | -5.98 (대폭 악화) |
| 2 | **+4.34** | **-2.96** | -7.30 (대폭 악화) |
| **3** | **-0.29** | **+4.43** | **+4.72 (큰 개선)** |
| 4 | +0.74 | +0.32 | -0.42 (약간 악화) |

**Fold 3의 Sharpe가 -0.29에서 +4.43으로 점프**. 1차 MVP가 망친 정확히 그 구간에서 Kalman이 명확히 작동.

## 4. 더 큰 발견 — 양면적 결과

가설은 지지됐지만, **전체 그림은 더 복잡**:

### 전체 페어 비교

| 페어 | 정적 μSharpe | 동적 μSharpe | 결과 |
|---|---|---|---|
| BAC~NVDA | +2.00 | -0.54 | 악화 |
| LUV~SLV | +1.52 | +0.00 | 악화 (Kelly=0) |
| GDX~GDXJ | +6.28 | -1.04 | **악화 (-7.32)** |
| PEP~V | 0.00 | +2.31 | 개선 (유일) |
| GDX~KO | +2.24 | +0.45 | 약화 |

**평균적으로 4/5 페어에서 Kalman이 정적보다 나쁨.**

## 5. 차트가 보여주는 진실 — `kalman_comparison_chart.png`

### Panel (1,1) — Beta trajectory
- 정적 β = 2.177 (평선)
- Kalman β = 0.8~1.1 사이 변동
- Train 종료 후 Kalman β 상승 (regime change 부분 포착 확인)

### Panel (1,2) — Spread comparison (가장 중요한 차트)
- **정적 spread**: -5.4 근처, 마지막 구간 -4.8로 시프트
- **동적 spread**: 0 근처에 거의 완벽한 안정
- → Kalman이 spread를 완벽하게 안정화시킴 (의도한 대로 작동)

### Panel (3,1) — Static vs Dynamic Mean Sharpe scatter
- 거의 모든 점이 y=x 라인 아래
- = Kalman 적용 후 평균적으로 Sharpe 악화

### Panel (3,2) — GDX~KO fold별 막대
- Fold 1, 2: 정적 압승 (큰 차이)
- Fold 3: 동적 압승 (큰 차이)
- → 구간별 완전히 다른 성격

## 6. 핵심 인사이트 — 왜 Kalman이 평균회귀를 망치는가

**Kalman은 spread를 안정화하도록 설계된 도구**다. 그런데 페어 트레이딩은 **spread가 불안정해야**(평균에서 일시 이탈해야) 거래 기회가 생긴다.

| 구간 유형 | 정적 베타 | Kalman |
|---|---|---|
| 정상 평균회귀 (fold 1, 2) | ✓ 잘 작동 | ✗ spread 흡수해 거래 사라짐 |
| Regime change (fold 3) | ✗ 망함 | ✓ 시프트 따라잡음 |

이건 명세서 v2 §7 실전 함정에 적은 **"노이즈를 베타 변화로 잘못 해석할 위험"**의 정확한 실증.

## 7. 두 도구의 본질적 충돌

| 도구 | 목적 | 작동 방식 |
|---|---|---|
| Kalman | 잠재 상태 추적 | spread를 0으로 수렴시킴 |
| 평균회귀 거래 | spread 일탈 활용 | spread의 분산이 클수록 좋음 |

두 도구가 **서로 반대 방향**으로 작동. Kalman을 켜면 평균회귀 거래가 약해지고, 정적 베타를 쓰면 regime change에 망함.

## 8. 다음 단계 — 어떻게 둘의 장점만 취할 것인가

### 후보 1. 선택적 Kalman 활성화
- 평상시: 정적 베타 사용
- Regime change 감지 시: Kalman 활성화
- 감지 방법:
  - CUSUM (누적합) on residuals
  - Bai-Perron 구조변화 검정
  - spread 평균의 rolling 시프트 크기

### 후보 2. Dual-strategy ensemble
- 정적 + 동적 두 전략을 동시 운용
- 자본 가중치는 최근 30일 Sharpe 비율로 조절
- 각 구간에서 잘 작동하는 쪽이 자연 우세

### 후보 3. 롤링 페어 재발굴 (원래 다음 후보였던 것)
- 30~60일마다 페어 자체를 재검정
- regime change로 페어 죽으면 자동 교체
- Kalman 없이도 본질적 해결

### 후보 4. Q/R 적응 튜닝
- 현재 delta=1e-5는 매우 보수적 (β가 천천히 변함)
- 더 작은 delta → 정적에 가까움 → fold 1, 2 개선
- 더 큰 delta → 빠른 적응 → fold 3 강화
- EM으로 데이터별 최적 Q/R 추정

## 9. 평가 — 가설 검증의 진짜 가치

이 실험의 가치는 가설 성공/실패 자체가 아니라:

1. **가설의 첫 부분(Fold 3 개선)은 명확히 입증**
2. **숨겨진 트레이드오프(Fold 1,2 악화) 발견**
3. **다음 우선순위 명확화**: Kalman만 단독으론 부족, 선택적/조건부 적용 필요

3단 검증 인프라(K-fold + Sensitivity + Permutation)가 없었다면 이 발견은 불가능. 단일 split만으로는 fold 1, 2의 악화를 못 봤을 것.

## 10. 운용 권장 — 현재 시점

| 페어 유형 | 권장 |
|---|---|
| 펀더멘털 합리적 + 안정된 통계 | **정적 베타** (예: GDX~GDXJ) |
| Regime change 위험 있는 페어 | **선택적 Kalman** (트리거 기반) |
| 미상 신규 페어 | **둘 다 백테스트 → ensemble** |

무조건 한쪽만 채택하는 건 위험.

## 11. 파일 구성

| 파일 | 역할 |
|---|---|
| `src/kalman_pairs.py` | Kalman β + 동적 spread + DynamicBacktester + K-fold |
| `scripts/run_kalman_test.py` | 정적 vs 동적 비교 + GDX~KO 상세 분석 |
| `charts/mvp4_kalman_comparison.png` | 3×2 패널 (베타/spread/boxplot×2/scatter/막대) |

## 12. 결론

> **가설 일부 지지 + 더 큰 발견.**

명세서 v2의 §7 Kalman은 **"regime change 보험"**으로는 명확히 작동. 하지만 **"표준 도구"**로는 부적합. 평균회귀 트레이딩과 본질적으로 충돌.

다음 단계: **롤링 페어 재발굴**이 가장 본질적 해결책. Kalman은 그 위에 옵션 모듈로 부착.
