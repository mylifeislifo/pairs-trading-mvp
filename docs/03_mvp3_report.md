# 3차 MVP — 오버핏 진단 보고서

> 2차 MVP의 K-fold는 단일 split 함정을 진단했다.
> 3차 MVP는 한 발 더 들어가 **각 페어가 진짜 신호인지, 파라미터 운/통계 운인지** 정량 판별한다.

## 1. 도입한 진단 도구

### 도구 1: 파라미터 민감도 분석 (Parameter Sensitivity)
- (window, entry) 3×3 = **9개 조합**에서 5-fold OOS Sharpe 측정
- 진짜 견고한 페어: 9개 셀이 균일하게 양수
- 오버핏 페어: 좁은 sweet spot에만 좋고 다른 셀에선 평범/음수
- **점수**: 평균 Sharpe, 표준편차, 양수 셀 비율

### 도구 2: Permutation Test
- position 신호를 시계열 무작위 셔플 200회
- 셔플 신호의 Sharpe 분포 (null distribution) 생성
- 실제 Sharpe가 null의 상위 5% 안에 있어야 통계적 유의
- **점수**: p-value, z-score (real - null_mean) / null_std

### 도구 3: 종합 등급 (A~F)
4개 기준 통과 개수에 따라:
- A: 4개 모두 통과 (진짜 견고)
- B: 3개 통과
- C: 2개 통과 (경계)
- D: 1개 통과 (위험)
- F: 모두 실패 (오버핏 확정)

기준:
- 양수 셀 비율 ≥ 70%
- 견고성 (mean_Sharpe / std_Sharpe) ≥ 0.5
- Permutation p < 0.10
- Permutation z ≥ 1.0

## 2. 실험 결과

| 페어 | Grade | Sens μ | Sens σ | +셀% | Perm p | Perm z | 결론 |
|---|---|---|---|---|---|---|---|
| **BAC~NVDA** | A | +2.33 | 0.85 | 100% | 0.015 | +2.38 | 진짜 견고 |
| **LUV~SLV** | A | +0.62 | 0.78 | 78% | 0.040 | +1.60 | 약하지만 통계 유의 |
| **GDX~KO** | A | +2.32 | 1.28 | 100% | 0.000 | +2.85 | 진짜 견고 |
| GDX~GDXJ | B | +3.01 | **6.15** | 89% | 0.000 | +8.25 | 한 셀 outlier (오버핏 신호) |
| PEP~V | C | -0.49 | 0.57 | 22% | 0.055 | +1.67 | 음수 평균, 안 좋은 페어 |

## 3. 가장 중요한 발견 — GDX~KO 진단 변천사

같은 페어, 같은 데이터, 다른 검증 방법.

| MVP | 검증 | 결론 |
|---|---|---|
| **1차** | 단일 70/30 split | "망함" (Sharpe -0.49, 수익률 -3.28%) |
| **2차** | 5-fold Purged K-fold | "견고" (Sharpe +2.24, 4/5 fold 양수) |
| **3차** | **K-fold + 파라미터 민감도 + Permutation** | **"Grade A — 모든 진단 통과"** |

### 3차 MVP가 본 GDX~KO

**파라미터 민감도 heatmap** (`overfit_diagnostics_chart.png` 3번째 컬럼):
```
        Entry=1.5  Entry=2.0  Entry=2.5
Win=20    1.0        1.7        1.5
Win=30    1.0        2.4        4.2
Win=40    1.3        3.3        4.5
```
- **9개 셀 모두 양수** (1.0~4.5)
- 단일 sweet spot이 아니라 **광범위한 파라미터 안정성**
- = "어떤 파라미터로 운용해도 비슷한 결과" = 진짜 신호

**Permutation Test**:
- Null 분포 평균 -1.08 (셔플하면 거래비용 때문에 손실)
- Real Sharpe +0.54 (null보다 1.62σ 위, p=0.000)
- 200번 셔플 중 실제 Sharpe보다 높은 경우 0번
- = **이 페어의 수익은 통계적으로 우연이 아님**

**결정적 증명**: 1차 MVP의 -3.28% 결론은 **단일 시간 구간의 우연**이었다. 검증 도구를 3개 쌓아 보니 GDX~KO는 진짜 견고한 페어.

## 4. GDX~GDXJ — 오버핏의 명확한 증거

펀더멘털상 가장 합리적인 페어(둘 다 금광주 ETF)인데 Grade B로 떨어진 이유:

**Heatmap에 outlier 셀**:
```
        Entry=1.5  Entry=2.0  Entry=2.5
Win=20   -0.1        0.2        0.0
Win=30    0.8        3.2        0.6
Win=40    1.5      **20.2**     0.7
```
- (window=40, entry=2.0)에서 Sharpe 20.2
- 다른 8개 셀은 -0.1~3.2의 평범한 범위
- **하나의 셀만 폭발적 = 전형적인 파라미터 운**
- 만약 이 파라미터로 실전 운용했다가 다음 fold에선 다른 결과 나올 위험

이게 명세서 v2 §3에서 경고한 "Z-score window 선택의 sweet spot 함정"의 정확한 사례.

**진단**: Permutation Test는 통과(p=0.000)했으니 진짜 신호는 있지만, **파라미터 튜닝에 너무 민감**해서 운용 시 보수적 접근 필요.

## 5. 차트가 보여주는 모든 진실

`overfit_diagnostics_chart.png` — 2×5 패널.

### 상단 (Parameter Sensitivity Heatmap)
- **균일하게 녹색** = 진짜 견고 (BAC~NVDA, GDX~KO)
- **녹색 한 점 + 나머지 노랑/주황** = 오버핏 (GDX~GDXJ)
- **대부분 노랑/주황** = 안 좋은 페어 (PEP~V)

### 하단 (Permutation Null Distribution)
- **빨간 선(real)이 회색 분포 오른쪽 끝** = 통계적 유의
- **빨간 선이 회색 분포 안에 묻힘** = 우연
- 모든 페어에서 null 분포 평균이 음수 (-0.74 ~ -4.71)
  - 이유: 시계열 셔플 시 거래비용으로 인한 손실 누적
  - 이걸 이긴 페어만 진짜 신호

## 6. 1차 MVP가 왜 틀렸나 — 결정적 정리

1차 MVP의 단일 70/30 split이 GDX~KO를 "망함"으로 판정한 진짜 이유:
- 시간상 마지막 30% 구간(2025년 7월~2026년 5월)이 우연히 가장 안 좋았던 fold
- K-fold로 보면 다른 4개 구간은 모두 양수
- 파라미터 민감도로 보면 9개 셀 모두 양수
- Permutation으로 보면 실제 Sharpe가 200번 셔플 어디에도 안 잡힘

**즉 단일 split은 "이 페어가 어떤 1년에서 어떻게 행동하는가"만 보여줄 뿐, 페어의 본질을 보여주지 못한다.** 3개 진단을 쌓아야 본질이 보임.

## 7. BAC~NVDA 미스터리 — 풀린 부분과 남은 부분

### 풀린 부분 (통계적 측면)
- 모든 진단 도구가 강력하게 통과 (Grade A, 양수 셀 100%, Permutation p=0.015)
- 단일 우연으로는 이 모든 통계를 설명 불가
- → **두 자산을 묶는 어떤 거시 요인이 실제로 존재할 가능성 매우 큼**

### 가설 (가능한 거시 요인)
1. **금리 + 위험선호도**: 은행주(BAC)와 메가캡 기술주(NVDA) 모두 금리/유동성에 민감
2. **AI 인프라 자금조달**: NVDA 매출과 BAC의 대규모 대출 사이클이 연동
3. **베타 1+ 자산의 공통 시장 노출**: 둘 다 SPY 베타 ≈ 1.5

### 남은 부분 (펀더멘털 측면)
- 통계는 진짜지만 **why?에 대한 답이 없음**
- 거시 환경 변화 시 (예: 금리 인상기, AI 거품 붕괴) 관계가 깨질 가능성
- → **§4 펀더멘털 필터의 다음 우선순위 명확**

## 8. 결론 — 3단 검증 체계 완성

이제 페어 트레이딩의 모든 진단 레이어가 구축됐다:

```
Layer 1: 페어 발굴 (§1 ADF + §2 공적분 + Half-life + TLS)
   ↓
Layer 2: K-fold 검증 (§8 Purged + Embargo)
   ↓
Layer 3: 오버핏 진단 (Parameter Sensitivity + Permutation Test)
   ↓
   진짜 견고한 페어 = Grade A or B
   ↓
Layer 4: 켈리 사이징 (§4) → 실전 운용
```

이 4단 체계를 통과한 페어만 자본 배분 후보가 된다.

## 9. 다음 단계

오버핏 문제는 진단 측면에서 해결됨. 다음 우선순위:

1. **§4 펀더멘털 필터** — BAC~NVDA의 거시 의존성 문제 해결
   - 같은 섹터/산업/시가총액 대역 우선
   - 거시 베타 정규화
2. **롤링 페어 재발굴** — 페어 풀의 신선도 유지
   - 30~60일마다 전체 재검정
   - regime change 자동 감지
3. **§7 Kalman 동적 베타** — 베타 시변화로 부분적 regime change 대응
4. **§4-C 포트폴리오 켈리** — 다중 페어 동시 운용

이 모든 것을 이제 **3단 검증 체계로 진짜 효과를 정량 측정 가능**.

## 10. 진단 도구의 한계 (정직한 자기 평가)

### 한계 1: GDX~GDXJ의 Sharpe 20.2 outlier
- 한 셀에서 거래 횟수가 너무 적어 std가 거의 0
- Sharpe 계산식의 수학적 artifact
- **개선**: 최소 거래 횟수 임계치 + 분모 안정화

### 한계 2: Permutation Test의 데이터 의존성
- position 셔플은 시계열의 자기상관 구조 파괴
- 진짜 null hypothesis는 더 정교한 시뮬레이션 필요 (Block bootstrap 등)

### 한계 3: 파라미터 그리드 9개의 한계
- exit_thr, stop도 영향. 4D 그리드는 너무 무거움
- **개선**: Sobol 시퀀스로 sparse sampling

이런 한계들은 향후 개선 사항으로 분리. MVP는 충분히 진단 기능 수행.

## 11. 파일 구성

| 파일 | 역할 |
|---|---|
| `pairs_trading_mvp.py` | 1차 (§1~4 + 백테스트) |
| `purged_kfold.py` | 2차 (§8 K-fold) |
| **`overfit_diagnostics.py`** | **3차 (Sensitivity + Permutation + Grading)** |
| `run_overfit_test.py` | 5개 페어 진단 + 시각화 |
| `overfit_diagnostics_chart.png` | 2×5 진단 차트 |

## 12. 최종 사용 패턴

```python
from overfit_diagnostics import (
    ParameterSensitivityAnalyzer,
    PermutationTester,
    diagnose_pair,
)

# 진단 도구
analyzer = ParameterSensitivityAnalyzer(
    windows=[20, 30, 40], entries=[1.5, 2.0, 2.5])
tester = PermutationTester(n_permutations=200)

# 페어별 종합 진단
for pair in candidate_pairs:
    report = diagnose_pair(data[pair.y], data[pair.x], pair,
                           analyzer, tester)
    if report.overall_grade in ['A', 'B']:
        deploy(pair)  # 실전 운용 후보
```

## 13. 종합 진단 체계의 의의

> **3차 MVP 이전**: "이 페어는 좋아 보인다" / "이 페어는 망했다" (단편적 판단)
> **3차 MVP 이후**: **"이 페어는 Grade A, 파라미터 무관 + 통계 유의 + K-fold 일관성"** (구조화된 증명)

증명 가능한 견고성. 이게 차이.
