# pairs-trading-mvp

> 통계적 차익거래 페어 트레이딩 — 알고리즘 명세서부터 4단 MVP까지

명세서 v2의 §1~4, §7, §8을 코드로 구현하고, 단일 split → K-fold → 오버핏 진단 → Kalman 동적 베타까지 검증 체계를 단계적으로 강화한 연구 기록.

## MVP 진화 요약

| MVP | 추가 도구 | 발견 |
|---|---|---|
| 1차 | 단일 70/30 split + 백테스트 엔진 | GDX~KO "망함" (-3.28%) |
| 2차 | Purged K-fold + Embargo | GDX~KO 사실은 "견고" (+2.24 Sharpe) |
| 3차 | Sensitivity + Permutation Test | GDX~KO "Grade A" 모든 진단 통과 |
| 4차 | Kalman 동적 베타 (§7) | 가설 부분 지지 — regime change 보험으로만 작동 |
| 5차 | Adaptive Ensemble | 안정성↑ 수익↓, 평가 metric 불일치 |
| 6차 | 롤링 페어 재발굴 | 영구 견고 페어 = 0개, 매월 50% 회전 |
| 7차 | 통합 시스템 + 4-way 비교 | **단순 고정 풀이 정교한 완전 통합을 압도** |
| 8차 | 품질 필터 (cap + min_sharpe) 튜닝 | 가설 기각, 손실 -20.86% → -6.53%로 축소. "Sharpe 필터는 잉여, cap만이 본질" |
| **9-A** | **(b) vs (d3) 격차 분해 — fixed_pair_pool 옵션** | **격차 -9.48%p의 -9.78%p가 "롤링 효과". 켈리/필터는 +0.30%p로 무해. 롤링 자체가 진짜 문제** |
| **9-B** | **신규 페어 cooldown 휴리스틱 sweep** | **가설 ✓ 입증. cd=30으로 -6.53% → +0.23% 양수 회복. 그러나 (b) +2.94%는 못 따라잡음. refresh_every=30이 자연 cooldown 단위** |
| **9-D** | **Refresh × Cooldown 2D grid sweep** | **★ (b) 최초 능가: refresh=180 + cd=0 → +6.54% (수익률), refresh=180 + cd=30 → Sharpe +0.73 / MDD -1.21%. but snapshot 3개로 표본 부족 경고. 회전율 자체가 본질적 원인임이 입증** |
| **9-E** | **10년 장기 데이터로 9-D 재검증** | **★★★ 9-D 결과 인공물 입증: 5개 변형 중 4개 Sharpe 부호 반전. 모든 변형 10년에서 음수. (b)조차 +2.94% → -0.55%로 무너짐. 진짜 결론: "3년 백테스트는 본질적으로 신뢰 불가". 6차 "영구 견고 페어=0개" 강력 재확인 (10년 train에서 2개만)** |
| **9-C** | **Walk-forward 검증 + 부트스트랩 95% CI** | **★★★ 통계적 사형 선고: 3개 변형 모두 alpha 없음 (CI에 0 포함 or 음수). (b) 양수 비율 64.3%지만 평균 -0.22 (꼬리 위험). r180,cd=30 양수 단 7.1% — 9-D는 완전 운빨로 확정. 메타 결론: "이 universe + 단순 z-score는 진짜 alpha 생성 불가"** |
| **10-A** | **Universe 확장 28→99 종목, walk-forward 재검증** | **★ 가설 기각: 평균 Sharpe -0.22 → -0.16 (+0.06 미미), 양수 비율 64.3% → 57.1% 감소, CI [-0.65, +0.32] 여전히 0 포함. 발견: 신호 양극화 (양수 윈도우는 더 양수, 음수는 더 음수). Universe 확장이 alpha 못 만들고 변동성만 키움. 페어 트레이딩 본질적 한계 확정** |

## 핵심 발견 모음

### Discovery 1 — 단일 split의 거짓말 (1차→2차→3차)

**같은 페어, 같은 데이터, 다른 검증 방법** :
- 1차: GDX~KO Sharpe **-0.49** ("망함")
- 2차: GDX~KO Sharpe **+2.24** ("견고")
- 3차: GDX~KO **Grade A** (3개 진단 모두 통과)

**1차 MVP의 단일 split이 우연히 가장 안 좋은 fold(fold 3)에 걸렸을 뿐**. K-fold + 오버핏 진단으로 보면 진짜 견고한 페어.

### Discovery 2 — Kalman의 양면성 (4차)

**가설 검증**: "Kalman 동적 베타가 fold 3 손실을 줄이나?"
- ✓ **Fold 3 Sharpe -0.29 → +4.43** (가설 지지)
- ✗ **Fold 1, 2의 Sharpe +5, +4 → -1, -3** (예상 못 한 악화)

**Kalman은 spread를 안정화시키는 도구. 그런데 평균회귀 거래는 spread 불안정을 활용**. 두 도구가 본질적으로 충돌.

**결론**: Kalman = "regime change 보험"으로는 명확히 작동, "표준 도구"로는 부적합. 선택적 활성화 필요.

## 디렉토리 구조

```
pairs-trading-mvp/
├── docs/             # 명세서 + 단계별 보고서
│   ├── 00_spec_v2.md         # 수학 알고리즘 명세서 (§1~10, v1 8개 오류 수정)
│   ├── 01_mvp1_report.md     # 1차 — 단일 split
│   ├── 02_mvp2_report.md     # 2차 — Purged K-fold
│   ├── 03_mvp3_report.md     # 3차 — 오버핏 진단
│   └── 04_mvp4_report.md     # 4차 — Kalman 동적 베타
├── src/              # 핵심 모듈
│   ├── pairs_trading_mvp.py  # §1~4: PairsFinder, SignalGenerator, KellySizer, Backtester
│   ├── purged_kfold.py       # §8: PurgedKFold + kfold_evaluate_pair
│   ├── overfit_diagnostics.py # Parameter Sensitivity + Permutation Test
│   └── kalman_pairs.py       # §7: 동적 베타 + 동적 spread + DynamicBacktester
├── scripts/          # 실행 스크립트
│   ├── run_backtest.py
│   ├── run_kfold_backtest.py
│   ├── run_overfit_test.py
│   └── run_kalman_test.py
└── charts/           # 시각화 결과
    ├── mvp1_pairs_backtest.png
    ├── mvp2_kfold_robustness.png
    ├── mvp3_overfit_diagnostics.png
    └── mvp4_kalman_comparison.png
```

## 4단 검증 체계

```
Layer 1: 발굴   (§1 ADF + §2 공적분 + Half-life + TLS)
Layer 2: 검증   (§8 Purged K-fold + Embargo)
Layer 3: 진단   (Parameter Sensitivity + Permutation Test + Grading A-F)
Layer 4: 베타   (§7 Kalman 동적 베타 — 선택적/조건부)
Layer 5: 배분   (§4 Kelly: 단일 + 포트폴리오)
```

## 실행

```bash
pip install numpy pandas scipy statsmodels yfinance matplotlib

python3 scripts/run_backtest.py        # 1차: 단일 split
python3 scripts/run_kfold_backtest.py  # 2차: K-fold
python3 scripts/run_overfit_test.py    # 3차: 오버핏 진단
python3 scripts/run_kalman_test.py     # 4차: Kalman 비교
```

## 시각화 모음

### 1차 — 단일 split 백테스트
![1차](charts/mvp1_pairs_backtest.png)

### 2차 — K-fold 견고성
![2차](charts/mvp2_kfold_robustness.png)

### 3차 — 오버핏 진단 (5개 페어)
![3차](charts/mvp3_overfit_diagnostics.png)

### 4차 — 정적 vs Kalman 동적
![4차](charts/mvp4_kalman_comparison.png)

### 5차 — Adaptive Ensemble
![5차](charts/mvp5_ensemble_comparison.png)

### 6차 — 롤링 페어 재발굴
![6차](charts/mvp6_rolling_pairs.png)

### 7차 — 통합 시스템 4-way 비교
![7차](charts/mvp7_production_comparison.png)

### 8차 — 품질 필터 튜닝 5-way 비교
![8차](charts/mvp8_tuned_comparison.png)

### 9차-A — (b) vs (d3) 격차 분해
![9-A](charts/mvp9a_decomposition.png)

### 9차-B — Cooldown sweep
![9-B](charts/mvp9b_cooldown_sweep.png)

### 9차-D — Refresh × Cooldown 2D sweep
![9-D](charts/mvp9d_refresh_sweep.png)

### 9차-E — 10년 장기 검증 (9-D 붕괴)
![9-E](charts/mvp9e_long_horizon.png)

### 9차-C — Walk-forward 통계적 검증
![9-C](charts/mvp9c_walkforward.png)

### 10차-A — Universe 확장 (28→99) 재검증
![10-A](charts/mvp10a_universe_expansion.png)

## 절대 모방 금지

명세서 v2 §10 / 부록 C에 명시:
- 만기일 호가 폭격 (LTP 누르기/띄우기): 시세조종, 형사처벌
- ETF 발행시장 직접 개입: AP 라이선스 필요
- 13F 종목 단순 추종: 임시 재고이므로 무의미
- 패시브 펀드 악의적 선행매매: 규제 회색지대

## 다음 단계

1. **롤링 페어 재발굴** — 가장 본질적 해결책 (regime change 자동 대응)
2. **선택적 Kalman 활성화** — CUSUM/Bai-Perron으로 구조변화 감지 시에만 켜기
3. **Dual-strategy ensemble** — 정적/동적 두 전략 동시 운용 후 가중
4. **§4 펀더멘털 필터** — 거시 의존성 거짓 양성 차단
5. **§4-C 포트폴리오 켈리** — 다중 페어 동시 운용

각 개선의 진짜 효과는 4단 검증 체계로 정량 비교 가능.

## 라이선스

연구 목적. 실전 운용 시 본인 책임.
