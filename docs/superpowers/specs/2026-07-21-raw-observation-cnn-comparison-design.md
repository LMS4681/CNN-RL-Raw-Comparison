# Raw Observation vs Candidate CNN Overnight Comparison Design

Date: 2026-07-21

## 1. Goal

현재 `CNN-RL`의 `candidate-cnn/full` 모델과, CNN 및 별도 구조화 특징
인코더를 제거한 `raw-direct/full` 모델을 같은 Colab 런타임에서 각각 약
3시간씩 순차 학습한다. 학습이 끝나면 Google Drive에 모델, 체크포인트,
평가 원자료, 그래프, 실행 환경 정보와 한국어 예비 비교 보고서를 남긴다.

이 실험은 다음 두 질문에 답한다.

1. 같은 약 3시간의 실제 실행 예산에서 어느 모델이 더 좋은 holdout 성능을
   얻는가?
2. 두 모델이 공통으로 도달한 timestep에서 CNN 기반 공간 표현이 원시
   구조화 관측의 직접 입력보다 나은가?

결과는 학습 seed 하나를 사용하는 단기 예비 실험이다. 통계적으로 확정적인
결론이나 CNN의 일반적 우월성을 주장하지 않는다.

## 2. Repository and Provenance

별도 공개 GitHub 저장소 이름은 `CNN-RL-Raw-Comparison`으로 한다. Colab에서
한 번의 clone만으로 실행할 수 있도록 기준 저장소 전체를 포함하는 독립
스냅샷으로 구성한다. 움직이는 upstream 브랜치나 submodule을 런타임에
따라가지 않는다.

기준 구현은 다음 커밋으로 고정한다.

```text
repository: https://github.com/LMS4681/CNN-RL.git
baseline commit: cd4e14fc1725a4ff159e59d6874d3602f3b65a06
observation schema: 3
fixed scenarios SHA256: 913cac9046dec8164ef65da60275522f7127de5ea775b1c5a6b6aac255716271
split manifest SHA256: 601bd6143ed8890577e5ff34921241d36fd6a0e99c4bdab4e26152ab168178f8
```

새 저장소는 `UPSTREAM_BASELINE.md`와 실행 manifest에 기준 URL, 전체 SHA,
고정 scenario 및 split-manifest SHA256, 비교 저장소 SHA를 기록한다. 모델,
TensorBoard event, ONNX 및 대형 체크포인트는 Git에 커밋하지 않고 Drive에
보관한다.

## 3. Compared Models

### 3.1 Candidate CNN arm

기준 모델의 candidate-CNN feature extractor는 현재 구현 그대로 사용한다.
SB3 2.9.0에서 현재 코드가 암묵적으로 생성하던 policy/value MLP 기본값을 두
arm 모두에 명시적으로 고정하므로, 이를 보고서에서는 `comparison-CNN arm`으로
부른다.

```text
extractor: candidate-cnn
state context: full
features dim: 256
```

이 모델은 10개 작업장 각각의 4채널 `64 x 64` 후보 배치 grid를 공유 CNN으로
인코딩한다. 현재 블록, 미래 블록, 미래 수요, pending queue는 기존 구조화
MLP를 통과한다. 작업장별 특징과 전체 작업장 특징도 기존 fusion MLP를
통과한다.

### 3.2 Raw direct arm

비교 모델은 parameter-free `RawDirectExtractor`를 사용한다. 다음 정규화된
schema-3 배열을 고정 순서로 mask 적용 후 평탄화하고 이어 붙인다.

```text
block
ws_meta
future_blocks
future_mask
future_demand
pending_blocks
pending_mask
pending_summary
```

이 순서의 raw feature dimension은 `2772`다. `future_mask`와 `pending_mask`는
유효하지 않은 slot을 0으로 만드는 데 사용하면서 관측값에도 그대로 포함한다.

`grids`는 모델 입력에서 제외한다. `ws_meta[:, 2]`는 이미 각 작업장의
`배치된 블록 면적 합 / 작업장 면적`이므로 별도 중복 특징을 만들지 않는다.
현재 및 미래 블록의 길이, 폭, 기간, 도착 시점, 형상비와 면적 등의 기존
정규화 값은 별도의 learned block encoder를 통과하지 않는다. invalid future
및 pending slot 값은 mask로 0으로 만든다.

공정한 단일 observation schema를 유지하기 위해 환경은 raw arm에서도 grid
배열을 생성하지만 extractor가 읽지 않는다. 따라서 처리속도 비교는 CNN forward
제거 효과를 포함하되 grid 생성 자체를 없앤 별도 환경 최적화 효과는 측정하지
않는다.

이 비교 모델에도 PPO가 행동 logits와 value를 계산하기 위해 사용하는 기본
policy/value MLP와 최종 선형 head는 남는다. 두 arm 모두 동일한 PPO 기본
policy/value MLP를 사용한다. 제거 대상은 CNN과 custom structured/fusion
feature-extractor MLP다.

SB3 버전 변화가 기본값을 바꾸지 못하도록 두 arm 모두 다음 contract를
명시한다.

```text
net_arch = {pi: [64, 64], vf: [64, 64]}
activation = ReLU
share_features_extractor = true
```

두 arm의 policy/value MLP topology는 같지만 입력 폭이 `256`과 `2772`로
다르므로 전체 parameter 수는 같지 않다.

### 3.3 Controlled and uncontrolled differences

두 arm에서 동일하게 유지할 항목은 데이터 split, 913-block episode, 작업장
순서, action mask, 무회전 배치 의미, 보상, 관측 정규화, full state context,
seed, PPO 하이퍼파라미터, holdout scenario와 런타임이다.

의도적으로 달라지는 항목은 feature extractor, 입력 차원, trainable parameter
수와 계산량이다. 따라서 보고서는 단순히 “CNN 유무만의 인과효과”라고 쓰지
않고, 현재 candidate CNN pipeline과 최소 raw-direct pipeline의 실용 성능
차이라고 기술한다. parameter-matched 추가 모델은 마감 이후 확장 과제로
남긴다.

## 4. Overnight Colab Workflow

노트북 하나가 같은 Colab Pro 런타임과 동일한 dependency lock 아래에서 arm별
격리 subprocess를 순차 실행한다. 각 subprocess가 기록한 VM boot ID, GPU UUID,
Torch/CUDA 버전과 lock hash가 일치하지 않으면 완결 결과로 표시하지 않는다.
실행 순서는 아래와 같다.

1. Drive mount 및 저장소 clone/정확한 비교 SHA checkout
2. 고정 dependency 설치와 GPU/CUDA/CPU/RAM 환경 기록
3. scenario 및 split-manifest hash 검증
4. 두 arm 각각 1,024-step save/load/evaluate smoke
5. `raw-direct/full`, seed 0을 기록 training-subprocess wall time 10,800초까지 학습
6. raw arm 선택 checkpoint 평가와 중간 보고서 저장
7. `candidate-cnn/full`, seed 0을 기록 training-subprocess wall time 10,800초까지 학습
8. CNN arm 선택 checkpoint 평가
9. 공통 timestep checkpoint 평가, 그래프 및 한국어 보고서 생성
10. artifact manifest와 `COMPLETE.json` 기록

새 모델을 먼저 실행해 비교 모델 결과를 우선 확보한다. 순서 효과와 단일 seed
한계는 보고서에 명시한다. smoke 중 하나라도 실패하면 장시간 학습을 시작하지
않고 오류 보고서만 Drive에 남긴다.

## 5. Time Budget and Fairness

각 arm의 목표 기록 training-subprocess wall time은 10,800초다. PPO rollout과
update, 주기적 holdout 선택, training callback 및 checkpoint I/O를 포함하며
초기 clone, dependency 설치, 두 smoke와 최종 평가는 제외한다. Colab 강제
종료 뒤에는 마지막 검증 checkpoint 이후 구간을 복원할 수 없으므로 이는
`정확히 3시간`이 아니라 checkpoint/heartbeat 간격만큼 오차가 생길 수 있는
best-effort 누적 예산이다. 목표, 기록 시간, end-to-end 시간, 재시작 횟수,
최대 미기록 가능 구간과 실제 초과분을 모두 보고한다.

시간 제한은 callback에서 monotonic clock으로 측정한다. 10,800초가 지난 뒤
처음 실행되는 environment callback에서 학습을 정상 종료한다. 진행 중인
holdout 평가가 있으면 그 평가를 강제로 중단하지 않으므로 wall-clock 초과분과
원인을 별도로 기록한다. timestep 10,000 또는 wall time 300초 중 먼저 도달한
시점마다 versioned checkpoint와 state를 갱신한다. 장시간 명령은 3시간보다 먼저
`learn()`이 끝나지 않도록 timestep ceiling `2,000,000,000`을 명시하며, callback
완료 전에 이 ceiling에 도달하면 실패로 처리한다. 다음 값을 `run_state.json`에
기록한다.

```text
target_training_seconds
completed_training_seconds
last_checkpoint_timestep
last_regular_checkpoint_timestep
last_checkpoint_file
last_checkpoint_sha256
config_sha256
generation
restart_count
max_unrecorded_seconds
status
started_at_utc
updated_at_utc
completed_at_utc
```

checkpoint는 로컬 unique temporary 파일에 저장하고 flush한 뒤 새 generation
이름으로 Drive에 복사한다. Drive에서 다시 열어 stored timestep과 SHA256을
검증한 뒤에만 unique temporary state와 replace로 `run_state.json`을 갱신한다.
Google Drive FUSE가 crash durability나 POSIX atomic rename을 보장한다고
가정하지 않으며, 검증 완료 generation만 재개 대상으로 인정한다. 재실행 시
`run_state.json`이 가리키는 exact checkpoint만 `--resume-from`으로 전달한다.
더 최신인 orphan archive는 자동 선택하지 않는다. 완료된 arm은 다시 학습하지
않는다.

동일 wall-clock 결과가 운영상 주 비교다. 표현 효과를 분리하기 위한 보조
비교는 두 arm이 모두 보유한 가장 큰 공통 10,000-step checkpoint에서
수행한다. 각 arm의 총 timestep과 steps/second도 함께 보고한다.

## 6. Fixed Training Contract

두 arm은 다음 값을 공유한다.

```text
state_context = full
learning_rate = 3e-4
n_steps = 960
batch_size = 64
n_epochs = 10
gamma = 1.0
gae_lambda = 0.98
seed = 0
n_envs = 1
vec_env = auto
device = auto
checkpoint_freq = 10000
holdout_eval_freq = 50000
holdout_selection_count = 5
```

candidate arm의 extractor output은 `256`, raw arm의 extractor output은 `2772`로
manifest에 별도 기록한다. 위 두 값은 서로 맞추지 않는다.

비교 저장소는 `requirements-comparison.txt`에 SB3, sb3-contrib, Gymnasium과
비GPU dependency를 정확한 버전으로 고정한다. Colab CUDA와 호환되는 사전 설치
Torch는 교체하지 않고 실제 Torch/CUDA/cuDNN 버전을 environment manifest에
기록한다. 두 arm은 같은 Colab VM에서 별도 Python subprocess로 실행하되 같은
dependency set을 사용한다. subprocess 격리는 첫 arm의 CUDA allocator와 Python
전역 상태가 둘째 arm에 남지 않게 한다.

검증 기준 버전은 `stable-baselines3==2.9.0`, `sb3-contrib==2.9.0`,
`gymnasium==1.3.0`이다. 나머지 non-GPU direct/transitive dependency는 구현
환경의 통과한 lock 파일로 고정하며 notebook은 그 lock 파일만 설치한다.

매 실행은 실제 resolved device, GPU 이름, Python, Torch, CUDA, cuDNN, 설치
패키지, 시작/종료 UTC, trainable parameter 수와 peak GPU memory를 기록한다.
두 arm이 같은 Colab VM과 GPU에서 순차 실행됐는지도 manifest로 검증한다.

## 7. Evaluation Contract

주기적 checkpoint 선택에는 fixed holdout seed `1000..1004`만 사용한다. 선택
기준은 terminal score가 높을수록, dropout이 낮을수록, delay가 낮을수록
우수한 기존 tuple 규칙을 유지한다.

최종 보고서의 주 성능은 선택에 사용하지 않은 seed `1005..1019`의 15개
scenario로 계산한다. 20개 전체 holdout 결과도 보조 표로 제공하되 validation
5개가 포함된 값임을 명시한다. original CSV 평가는 business reference일
뿐 일반화 성능으로 해석하지 않는다.

각 arm에 대해 다음을 저장한다.

- terminal score, dropout rate, mean delay와 delayed count;
- scenario별 원자료;
- 학습 timestep 및 wall-clock 곡선;
- PPO loss 및 episode 지표;
- trainable parameter 수;
- 총 timestep, steps/second, 평가 시간, peak GPU memory;
- best/final/common-step checkpoint 식별자와 SHA256.

3시간 주 결과는 best model이 있으면 `3시간 budget 안에서 validation 5개로
선택된 checkpoint`라고 표현하고, 없으면 `주기적 선택 전에 종료되어 마지막
검증 checkpoint를 사용한 fallback`이라고 표현한다. checkpoint timestep,
선택 횟수와 선택 점수 또는 fallback 사유를 함께 쓴다.
scenario paired CSV에는 terminal score뿐 아니라 dropout, mean delay와 delayed
count의 CNN-minus-raw 차이를 모두 기록한다.

한 arm에 `best_model`이 생성되기 전에 시간 제한에 도달하면 마지막 readable
training checkpoint를 평가하고 보고서에 fallback 사실을 표시한다.

## 8. Google Drive Artifacts

기본 저장 경로는 다음과 같다.

```text
/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721/
```

구조는 다음과 같다.

```text
manifest.json
environment.json
stage_journal.json
raw_direct/
  run_state.json
  run_origin.json
  training_completion.json
  run_config.json
  environment_segments.jsonl
  runtime_metrics.json
  progress_timing.csv
  checkpoints/
  best_model.sb3
  block_placement_ppo.sb3
  holdout_selection.csv
  evaluation_scenarios.csv
  evaluation_primary_test.csv
  training_log.csv
  loss_log.csv
candidate_cnn/
  ...same contract...
comparison/
  common_step_evaluation.csv
  scenario_paired_differences.csv
  summary.json
  learning_curves.png
  holdout_comparison.png
  preliminary_comparison_ko.md
COMPLETE.json
```

`COMPLETE.json`은 두 smoke, 두 time budget, 두 평가와 report generation이 모두
성공하고 다음 무결성 gate를 통과한 뒤에만 생성한다: baseline/comparison/config/
scenario/split/lock hash 일치, 두 arm의 VM boot ID와 GPU UUID 일치, state가
가리키는 checkpoint의 hash와 stored timestep 일치, selection seed 5개와
primary-test seed 15개의 정확한 분리 및 row 중복 없음. 중간 실패 시 가능한
stage 경계에서 `PARTIAL_REPORT.md`와 마지막 성공 단계, 재개 명령을 남긴다.
VM 강제 종료로 파일을 쓸 기회가 없었던 경우 다음 실행 시작 시 stale
`in_progress` 단계를 `interrupted`로 바꾸고 같은 입력 hash의 완료 stage만
건너뛴다.

## 9. Korean Preliminary Report

자동 생성 보고서는 다음 순서로 작성한다.

1. 실험 목적과 두 pipeline 설명
2. 통제 변수와 의도적으로 다른 변수
3. Colab 하드웨어 및 실행 시간
4. 3시간 budget 내 validation-selected checkpoint 성능 비교
5. 공통 timestep 성능 비교
6. 학습곡선, 처리속도와 parameter 수 비교
7. 15개 미사용 holdout scenario의 paired 결과
8. 20개 전체 holdout 보조 결과
9. 해석: 공간 topology 정보와 raw scalar 정보의 장단점
10. 제한: seed 1개, 순차 실행 순서, 동적 Colab 자원, parameter mismatch
11. 다음 실험: paired seeds 1~4 및 parameter-matched control

보고서 표현은 “승패 확정”이 아니라 “seed 0의 3시간 budget 내 validation-selected
checkpoint 예비 결과”로 제한한다. UTF-8로 생성하고 replacement character가
포함되면 완결 보고서 생성을 실패시킨다.
원자료가 불완전하면 수치를 추정하지 않고 누락 원인과 재개 방법을 쓴다.

## 10. Repository Layout

```text
CNN-RL-Raw-Comparison/
  AllocRL/                         # pinned baseline snapshot
    comparison/
      raw_direct_extractor.py
      wall_clock_callback.py
      experiment_runner.py
      report_builder.py
      artifact_manifest.py
    configs/overnight_seed0.json
    test_raw_direct_extractor.py
    test_wall_clock_callback.py
    test_comparison_experiment.py
    test_comparison_report.py
  notebooks/overnight_compare.ipynb
  reports/README.md
  UPSTREAM_BASELINE.md
  README.md
  .gitignore
```

비교 관련 새 코드는 `AllocRL/comparison/`에 경계를 두고 기존 환경·평가 contract를
재사용한다. 기존 baseline 동작을 비교 편의를 위해 바꾸지 않는다.

## 11. Verification

구현 전 테스트는 다음 행위를 먼저 실패로 증명한다.

- raw extractor가 grid를 무시하고 고정 순서의 mask 적용 raw vector를 반환;
- raw extractor 내부에 `Conv2d` 또는 `Linear`가 없음;
- 두 arm의 PPO policy/value MLP contract가 동일;
- fake clock에서 누적 10,800초에 정지하고 resume 시 남은 시간만 실행;
- selection 평가 시간이 동일 예산에 포함됨;
- state가 가리키는 exact archive 재개, orphan/corrupt checkpoint 처리;
- 300초 heartbeat, 완료된 arm 재실행 방지와 stage별 중단 복구;
- selection seed와 final test seed 분리;
- parameter/runtime/environment manifest 기록;
- 보고서가 실제 CSV만 사용하고 누락값을 만들어내지 않음;
- notebook JSON 유효성, 비어 있는 output cell, Drive 경로와 순차 순서;
- 두 arm의 1,024-step save/load/evaluate smoke.

구현 완료 조건은 focused tests, 기존 전체 regression, notebook contract 검사,
두 smoke와 짧은 로컬 time-boxed end-to-end가 모두 통과하는 것이다. 실제
6시간 결과는 코드 완료 조건이 아니라 Colab 운영 단계의 산출물이다.

### 11.1 Persistence hardening clarification

`run_state.json`의 `complete`는 학습 budget/checkpoint commit만 뜻하며 train stage
전체 완료를 뜻하지 않는다. archive 검증 뒤 state를 먼저 원자적으로 쓰고 다시 읽은
다음, 전체 `progress_timing.csv`를 원자적으로 교체한다. 재시작 시 progress는 state와
엄격히 reconcile하며 malformed/conflicting evidence는 삭제하지 않고 실패한다. 모델 생성
뒤 첫 `learn()` 전 관측 timestep은 `run_origin.json`에 기록한다. 평가, conventional 모델,
runtime metrics가 모두 검증된 뒤에만 마지막으로 `training_completion.json` receipt를 쓰며,
runner의 train-stage 완료/skip 기준은 이 receipt이다. 같은 GPU model/memory 및
Torch/CUDA/cuDNN/runtime 사실을 유지한 새 VM boot/GPU instance는 재개할 수 있고,
boot ID와 GPU UUID 자체는 segment provenance로 보존하되 cross-segment equality 조건은 아니다.

## 12. Operational Limitations

Colab은 GPU 종류, runtime lifetime과 사용 한도를 보장하지 않는다. 이 설계는
정상적인 한 런타임 완주와 마지막 검증 generation부터의 수동 재실행 복구를
지원하지만, 마지막 heartbeat 이후 사용 시간은 복구할 수 없고 사용자가 자는
동안 강제 종료된 런타임을 자동으로 다시 연결할 수도 없다.
따라서 다음 날 `COMPLETE.json`이 없으면 `PARTIAL_REPORT.md`의 재개 셀을 한 번
실행해야 한다.

이 제한 때문에 결과 문서는 Drive에 가능한 범위까지 항상 생성하되, 완료되지
않은 arm을 완료된 것처럼 비교하지 않는다.
