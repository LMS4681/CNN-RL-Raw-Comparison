# AllocRL A-E Ablation

이 실험은 같은 학습 예산과 같은 고정 평가 시나리오에서 미래 블록 정보와
후보 배치 CNN의 효과를 분리해 비교합니다.

모든 학습군은 동일한 데이터 정책을 사용합니다. 7월과 11월을 제외한 913개
블록을 지정된 10개 빈 작업장에 배치하며, 에피소드 총량은 913개로 고정합니다.
학습 시 월별 평준화 랜덤 프로필과 실제 월별 프로필을 80:20으로 혼합하고,
고정 평가는 실제 월별 프로필과 고정 작업장 geometry를 사용합니다.

## 실험군

| ID | Extractor | Future blocks | 비교 목적 |
| --- | --- | ---: | --- |
| A | `structured` | 0 | 현재 상태만 사용하는 최소 기준선 |
| B | `structured` | 4 | 순서가 보존된 미래 블록 정보의 효과 |
| C | `fixed-grid` | 4 | 학습하지 않는 직접 이미지 입력의 효과 |
| D | `candidate-cnn` | 0 | 학습된 공간 특징만의 효과 |
| E | `candidate-cnn` | 4 | 권장 전체 모델 |

## 실행 순서

고정 평가 파일은 한 번만 생성합니다. 이 파일에는 학습 seed와 분리된 평가
seed `1000..1019`가 들어갑니다.

```powershell
py -B run_ablation.py --prepare-eval-scenarios
```

먼저 screening 명령 15개(5개 모델 x seed 3개)를 확인하고 실행합니다.

```powershell
py -B run_ablation.py --mode screening --dry-run
py -B run_ablation.py --mode screening
```

screening 결과를 확인한 뒤 최종 비교 25개(5개 모델 x seed 5개)를 실행합니다.

```powershell
py -B run_ablation.py --mode final --dry-run
py -B run_ablation.py --mode final
```

각 실행은 `output_ablation/<mode>/<ID>/seed_<seed>`에 모델과 로그를 저장합니다.
회사 보안 환경에서 `.zip`이 변환되는 문제를 피하기 위해 SB3 모델과
체크포인트 확장자는 `.sb3`입니다.

## 평가 지표

우선순위는 다음과 같습니다.

1. `mean_terminal_score`
2. `mean_dropout_rate`
3. `mean_delay_days`, `mean_delayed_count`
4. seed별 학습 속도와 분산
5. `mean_retained_choice_ratio`

retained-choice ratio는 현재 행동 직전에 고정한 다음 K개 블록에 대해 가능한
작업장 선택 수를 세고, 행동 직후 같은 블록 집합을 다시 계산한
`after / before`입니다. `before=after=0`이면 `1.0`으로 기록합니다. 이 값은
평가 전용이며 관측, 보상, action mask에 들어가지 않습니다.

원본 CSV 평가는 `evaluation_csv.csv`, 고정 시나리오 평가는
`evaluation_scenarios.csv`에 별도로 기록됩니다.

## 승인 기준

E가 B보다 다음 중 하나를 만족해야 합니다.

- 평균 terminal score 절대 개선이 `0.05` 이상
- dropout rate 상대 감소가 `10%` 이상

또한 개선 방향이 최종 seed 5개 중 최소 4개에서 일관되어야 합니다. E와 C는
학습된 convolution의 효과를, E와 D는 미래 블록 정보의 효과를 확인합니다.
