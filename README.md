# pairs-trading-mvp

> 통계적 차익거래 페어 트레이딩 — 알고리즘 명세서부터 3단 검증 체계까지

명세서 v2의 §1~4, §8을 코드로 구현하고, 단일 split → K-fold → 오버핏 진단으로 검증 체계를 단계적으로 강화한 연구 기록.

## 핵심 발견

**GDX~KO 페어 진단의 3단 변천사**:

| MVP | 검증 방법 | 결론 |
|---|---|---|
| 1차 | 단일 70/30 split | "망함" (Sharpe -0.49, 수익률 -3.28%) |
| 2차 | 5-fold Purged K-fold | "견고" (Sharpe +2.24, 4/5 fold 양수) |
| **3차** | **+ Sensitivity + Permutation** | **"Grade A — 모든 진단 통과"** |

같은 페어, 같은 데이터. 다른 건 검증 방법뿐. **1차 MVP가 완전히 잘못된 결론을 내렸음**이 결정적으로 증명됨.

## 디렉토리 구조

```
pairs-trading-mvp/
├── docs/             # 명세서 + 단계별 보고서
│   ├── 00_spec_v2.md         # 수학 알고리즘 명세서 (§1~10, v1 8개 오류 수정)
│   ├── 01_mvp1_report.md     # 1차 — 단일 split 백테스트
│   ├── 02_mvp2_report.md     # 2차 — Purged K-fold
│   └── 03_mvp3_report.md     # 3차 — 오버핏 진단
├── src/              # 핵심 모듈
│   ├── pairs_trading_mvp.py  # PairsFinder, SignalGenerator, KellySizer, Backtester
│   ├── purged_kfold.py       # PurgedKFold + kfold_evaluate_pair
│   └── overfit_diagnostics.py # ParameterSensitivityAnalyzer + PermutationTester
├── scripts/          # 실행 스크립트
│   ├── run_backtest.py
│   ├── run_kfold_backtest.py
│   └── run_overfit_test.py
└── charts/           # 시각화 결과
    ├── mvp1_pairs_backtest.png
    ├── mvp2_kfold_robustness.png
    └── mvp3_overfit_diagnostics.png
```

## 3단 검증 체계

```
Layer 1: 발굴   (ADF + 공적분 + Half-life + TLS)
Layer 2: 검증   (Purged K-fold + Embargo)
Layer 3: 진단   (Parameter Sensitivity + Permutation Test + Grading A-F)
Layer 4: 배분   (Kelly: 단일 + 포트폴리오)
```

## 실행

```bash
pip install numpy pandas scipy statsmodels yfinance matplotlib

# 1차: 단일 split 백테스트
python3 scripts/run_backtest.py

# 2차: K-fold 검증
python3 scripts/run_kfold_backtest.py

# 3차: 오버핏 진단
python3 scripts/run_overfit_test.py
```

## 명세서 v2 — 구현된 알고리즘

`docs/00_spec_v2.md` 참조. 핵심 항목:

| § | 알고리즘 | 구현 위치 |
|---|---|---|
| 1 | ADF 단위근 검정 | `src/pairs_trading_mvp.py::PairsFinder.is_nonstationary` |
| 2 | 공적분 + Half-life + TLS | `src/pairs_trading_mvp.py::PairsFinder.screen_pairs` |
| 3 | Z-score (position + force_close) | `src/pairs_trading_mvp.py::SignalGenerator` |
| 4 | 켈리 (단일 + 포트폴리오) | `src/pairs_trading_mvp.py::KellySizer` |
| 8 | Purged K-fold + Embargo | `src/purged_kfold.py::PurgedKFold` |
| - | Parameter Sensitivity | `src/overfit_diagnostics.py::ParameterSensitivityAnalyzer` |
| - | Permutation Test | `src/overfit_diagnostics.py::PermutationTester` |

## 시각화 결과

### 1차 MVP — 단일 split 백테스트
![1차](charts/mvp1_pairs_backtest.png)

### 2차 MVP — K-fold 견고성
![2차](charts/mvp2_kfold_robustness.png)

### 3차 MVP — 오버핏 진단
![3차](charts/mvp3_overfit_diagnostics.png)

## 절대 모방 금지

명세서 v2 §10 / 부록 C에 명시:
- 만기일 호가 폭격 (LTP 누르기/띄우기): 시세조종, 형사처벌
- ETF 발행시장 직접 개입: AP 라이선스 필요
- 13F 종목 단순 추종: 임시 재고이므로 무의미
- 패시브 펀드 악의적 선행매매: 규제 회색지대

## 다음 단계

1. **§4 펀더멘털 필터** — BAC~NVDA 같은 거시 의존성 거짓 양성 차단
2. **롤링 페어 재발굴** — 페어 풀 신선도 유지 (30~60일 주기)
3. **§7 Kalman 동적 베타** — regime change 부분 대응
4. **§4-C 포트폴리오 켈리** — 다중 페어 동시 운용

각 개선의 진짜 효과는 이제 3단 검증 체계로 정량 비교 가능.

## 라이선스

연구 목적. 실전 운용 시 본인 책임.
