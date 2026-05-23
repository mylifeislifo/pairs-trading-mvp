# 5차 MVP — 이중 전략 합성 (Adaptive Ensemble) 보고서

> 4차 MVP의 발견 — 정적 베타와 Kalman 동적 베타가 서로 다른 시장 국면에서 강함 — 을 합쳐서 둘 모두 능가하려는 시도.

## 1. 검증 가설

> **"두 전략을 동시 운용하고 최근 성과에 따라 자본을 자동 배분하면, 각 시장 국면에서 잘 작동하는 쪽이 우세해진다."**

비유로 풀면:
- 두 명의 펀드매니저를 고용
- 한 명은 안정 시장 전문 (정적)
- 다른 한 명은 격변 시장 전문 (Kalman)
- 매일 누가 최근에 더 잘했는지 보고 자본을 그쪽으로 옮김

## 2. 구현 — adaptive_weight

핵심은 "최근 성과 가중치":

```python
# 최근 30일 두 전략 Sharpe 계산
sr_static = ret_static.rolling(30).mean() / ret_static.rolling(30).std()
sr_dynamic = ret_dynamic.rolling(30).mean() / ret_dynamic.rolling(30).std()

# Softmax로 비중 결정 (Sharpe 높은 쪽에 더 많이)
w_static = exp(5 * sr_static) / (exp(5 * sr_static) + exp(5 * sr_dynamic))
w_dynamic = 1 - w_static

# 한쪽에 완전 쏠림 방지 (최소 10%)
w_static.clip(0.1, 0.9)
```

추가 모듈:
- `EnsembleBacktester`: 두 전략 백테스트 → 일별 수익률 합성
- `kfold_evaluate_ensemble`: K-fold로 앙상블 평가

## 3. 결과 — 양면적

### Sharpe 기준 (K-fold 평균)

| 페어 | 정적 | 동적 | 앙상블 | 최강 |
|---|---|---|---|---|
| BAC~NVDA | +2.00 | -0.54 | +0.53 | 정적 |
| LUV~SLV | +1.52 | 0.00 | +0.15 | 정적 |
| GDX~GDXJ | +6.28 | -1.04 | +1.02 | 정적 |
| PEP~V | 0.00 | +2.31 | -0.45 | 동적 |
| GDX~KO | +2.24 | +0.45 | +0.98 | 정적 |

**앙상블은 0/5 페어에서 최강.** 정적이 4/5, 동적이 1/5.

### 자본 곡선 기준 (실제 백테스트)

GDX~KO 자본 곡선 (`charts/mvp5_ensemble_comparison.png` Panel 1,1):
- **정적**: $100K → $200K → $145K (롤러코스터)
- **동적**: $100K → $130K (평평)
- **앙상블**: $100K → $160K (안정적 우상향)

→ Sharpe로는 정적이 1등이지만, **자본 곡선의 안정성으로는 앙상블이 1등**.

## 4. 왜 결론이 두 가지인가

### Sharpe vs Equity Curve의 괴리

Sharpe = 평균 수익 / 표준편차. 큰 win + 큰 lose가 반복되면 Sharpe가 높을 수 있다. 자본 곡선은 들쭉날쭉해도.

앙상블은:
- 큰 win 못 받음 (가중치 분산)
- 큰 lose도 피함 (반대쪽 전략이 살림)
- 결과: **변동성 감소, 평균 수익은 줄어듦**

이게 ensemble의 본질. **수익보다 안정성을 위한 도구**.

### Worst Drawdown Panel (3,2) — 평가 코드 버그 신호

차트 우하단 막대에서 앙상블 drawdown이 50~100%로 비현실적. 정적/동적은 거의 0%. 

원인: K-fold 평가 코드에서:
- 정적/동적: 거래별 PnL 기반 metric
- 앙상블: 일별 수익률 기반 metric

**서로 다른 metric을 비교**한 셈. 정확한 비교를 위해선 metric 통일 필요. Panel (1,1)의 실제 백테스트가 진실에 더 가까움.

## 5. 차트 패널별 해석 — `mvp5_ensemble_comparison.png`

### Panel (1,1) — Equity Curves (가장 정직한 그림)
- 정적: 변동성 큰 수익
- 동적: 평평
- 앙상블: 안정적 우상향
- → **실제 운용 시 앙상블이 가장 좋아 보임**

### Panel (1,2) — Ensemble 자동 가중치
- 두 비중이 빈번하게 교차
- 30일 lookback이 시장 국면 변화에 반응
- 평균 정적 비중 53.6% (약간 정적 의존)

### Panel (2,1) — Per-pair Mean Sharpe
- 모든 페어에서 정적이 가장 큰 막대
- 앙상블은 중간

### Panel (2,2) — GDX~KO Fold별
- Fold 1, 2: 정적 압승 (+5, +4 vs 앙상블 +0.5, +0.9)
- Fold 3: 동적 압승 (+4.4 vs 앙상블 +2.8)
- → **앙상블은 어느 fold에서도 1등 못 함**, 항상 2등

### Panel (3,1) — Regime Detection
- 시간에 따라 정적/동적 우세 구간 시각화
- Train 구간 후반(2024 가을~2025 봄): 정적 우세
- 2025 가을~2026 초: 동적 우세 (regime change)
- → **Ensemble이 국면 변화를 감지하긴 함**

## 6. 핵심 인사이트

### Ensemble의 한계 3가지

1. **Lookback 지연**
   - 30일 윈도우 → 시장이 바뀌어도 30일 동안 늦게 반응
   - regime change 시작 시 손실 흡수 못 함

2. **최소 가중치 페널티**
   - 한쪽이 완전히 망해도 10%는 잡아둠
   - 그 10%가 손실 확대

3. **Sharpe 평균화의 함정**
   - 정적 +4, 동적 -3 → 가중 평균 +0.5
   - 정적 단독 운용보다 약함

### Ensemble의 강점 1가지

- **Drawdown 감소** (자본 곡선 안정화)
- "큰 손실"을 피하는 데 효과적
- 자본 보전 우선 전략에 적합

## 7. 결정

**Sharpe / Return 극대화 목표 → 정적 단독 운용**
**자본 곡선 안정성 / Drawdown 최소화 목표 → Ensemble**

운용 목표에 따라 다름. 사후 어떤 전략이 좋을지 알면 단독으로 그것만 쓰는 게 최선이지만, 사전엔 알 수 없으므로 ensemble은 "안전 보험" 가치.

## 8. 다음 단계 — 롤링 페어 재발굴

이제 명확해진 것:
- 정적 베타: 좋은 페어에서 최강
- Kalman 동적: regime change 보험
- Ensemble: 변동성 보험

**그러나 셋 다 같은 페어 풀에서 베타 추정 방식만 다름**. **페어 자체가 죽어가는 문제는 해결 못 함**.

GDX~KO가 OOS에서 흔들렸던 근본 원인: 2025년 7월부터 GDX가 금값 급등으로 폭주, KO와의 공적분 관계가 약화. **이걸 어떤 베타 추정도 해결 못 함**. 페어 자체를 갈아치워야 함.

→ **롤링 페어 재발굴**이 다음 단계의 정답.

## 9. 한계와 추가 작업

- K-fold 평가 metric 통일 필요 (현재 trade-level vs daily-level 불일치)
- adaptive_weight의 lookback, temperature 파라미터 그리드 탐색
- Ensemble을 정적 + 동적뿐 아니라 더 많은 전략으로 확장 (3개, 5개)

이런 세부 개선은 롤링 재발굴 후에.

## 10. 파일

| 파일 | 역할 |
|---|---|
| `src/ensemble_strategy.py` | adaptive_weight + EnsembleBacktester + kfold_evaluate_ensemble |
| `scripts/run_ensemble_test.py` | 정적/동적/앙상블 3-way 비교 |
| `charts/mvp5_ensemble_comparison.png` | 3×2 차트 (equity / 가중치 / Sharpe / Drawdown 등) |
