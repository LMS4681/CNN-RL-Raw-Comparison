"""Build deterministic, single-seed preliminary comparison report artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparison.checkpoint_evaluator import ARMS, EVALUATION_COLUMNS, PRIMARY_TEST_SEEDS, SELECTION_SEEDS


HEADLINE_COLUMNS = (
    "mean_terminal_score", "mean_dropout_rate", "mean_delay_days", "mean_delayed_count",
)
PAIR_COLUMNS = (
    "seed", "terminal_score_delta_cnn_minus_raw", "dropout_rate_delta_cnn_minus_raw",
    "mean_delay_days_delta_cnn_minus_raw", "delayed_count_delta_cnn_minus_raw",
)
MISSING = "자료 없음"
RUNTIME_FIELDS = (
    "target_training_seconds", "recorded_training_seconds", "end_to_end_training_seconds",
    "overrun_seconds", "restart_count", "max_unrecorded_seconds", "start_timestep",
    "end_timestep", "steps_per_second", "parameter_counts", "peak_cuda_memory_bytes",
    "evaluation_seconds", "selected_checkpoint_timestep", "selection_count",
    "selection_tuple", "checkpoint_identity",
)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"missing canonical artifact: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _read_evaluation(path: Path, arm: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError(f"evaluation CSV has incompatible header: {path}")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"missing evaluation CSV: {path}") from error
    if len(rows) != len(seeds):
        raise ValueError(f"evaluation CSV must contain exact {len(seeds)} rows: {path}")
    expected_partition = "selection" if seeds == SELECTION_SEEDS else "primary_test"
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != set(EVALUATION_COLUMNS) or row["arm"] != arm:
            raise ValueError(f"evaluation CSV has invalid arm/schema: {path}")
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"evaluation seed must be an integer: {path}") from error
        row_partition = "selection" if seed in SELECTION_SEEDS else "primary_test"
        if row["evaluation_partition"] != row_partition:
            raise ValueError(f"evaluation partition is invalid: {path}")
        converted = dict(row); converted["seed"] = seed
        for column in HEADLINE_COLUMNS:
            converted[column] = _number(row[column], column)
        normalized.append(converted)
    if tuple(sorted(row["seed"] for row in normalized)) != seeds:
        raise ValueError(f"evaluation CSV must contain exact unique {expected_partition} seeds: {path}")
    return sorted(normalized, key=lambda row: row["seed"])


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    arm_root = root / arm
    all_rows = _read_evaluation(arm_root / "evaluation_scenarios.csv", arm, tuple(range(1000, 1020)))
    selection = [row for row in all_rows if row["seed"] in SELECTION_SEEDS]
    primary = _read_evaluation(arm_root / "evaluation_primary_test.csv", arm, PRIMARY_TEST_SEEDS)
    all_primary = [row for row in all_rows if row["seed"] in PRIMARY_TEST_SEEDS]
    if any(any(row[column] != corresponding[column] for column in HEADLINE_COLUMNS) for row, corresponding in zip(primary, all_primary)):
        raise ValueError(f"primary CSV disagrees with all-scenarios CSV: {arm}")
    return {"selection_rows": selection, "primary_rows": primary, "all_rows": all_rows, "runtime": _runtime(arm_root / "runtime_metrics.json")}


def _runtime(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    data = {field: data.get(field) for field in RUNTIME_FIELDS}
    counts = data.get("parameter_counts")
    if counts is not None:
        if not isinstance(counts, Mapping) or set(counts) != {"total", "feature_extractor", "policy", "value"}:
            raise ValueError("parameter_counts must contain exactly total, feature_extractor, policy, value")
        parsed = {key: int(_number(value, f"parameter_counts.{key}")) for key, value in counts.items()}
        if any(value < 0 for value in parsed.values()) or parsed["total"] != parsed["feature_extractor"] + parsed["policy"] + parsed["value"]:
            raise ValueError("disjoint parameter counts do not reconcile")
        data = dict(data); data["parameter_counts"] = parsed
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"runtime metric {key} must be finite")
    return data


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {f"mean_{column.removeprefix('mean_')}": sum(row[column] for row in rows) / len(rows) for column in HEADLINE_COLUMNS}


def build_paired_differences(root: str | Path) -> list[dict]:
    """Return primary-test scenario deltas, always candidate CNN minus raw direct."""
    base = Path(root)
    raw = _load_arm(base, "raw_direct")["primary_rows"]
    cnn = _load_arm(base, "candidate_cnn")["primary_rows"]
    by_raw = {row["seed"]: row for row in raw}; by_cnn = {row["seed"]: row for row in cnn}
    if set(by_raw) != set(PRIMARY_TEST_SEEDS) or set(by_cnn) != set(PRIMARY_TEST_SEEDS):
        raise ValueError("primary-test pairing requires exact matching seeds")
    return [{
        "seed": seed,
        "terminal_score_delta_cnn_minus_raw": by_cnn[seed]["mean_terminal_score"] - by_raw[seed]["mean_terminal_score"],
        "dropout_rate_delta_cnn_minus_raw": by_cnn[seed]["mean_dropout_rate"] - by_raw[seed]["mean_dropout_rate"],
        "mean_delay_days_delta_cnn_minus_raw": by_cnn[seed]["mean_delay_days"] - by_raw[seed]["mean_delay_days"],
        "delayed_count_delta_cnn_minus_raw": by_cnn[seed]["mean_delayed_count"] - by_raw[seed]["mean_delayed_count"],
    } for seed in PRIMARY_TEST_SEEDS]


def _common_summary(root: Path) -> dict[str, Any]:
    path = root / "comparison" / "common_step_evaluation.csv"
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError("common-step CSV has incompatible header")
            rows = list(reader)
    except OSError as error:
        raise ValueError("missing common-step evaluation CSV") from error
    if len(rows) != 40:
        raise ValueError("common-step CSV must have exactly 40 rows")
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        arm = row.get("arm")
        if set(row) != set(EVALUATION_COLUMNS) or arm not in grouped or row.get("checkpoint") != "common_step":
            raise ValueError("common-step CSV has invalid arm/checkpoint")
        normalized = dict(row); normalized["seed"] = int(row["seed"])
        if normalized["evaluation_partition"] != ("selection" if normalized["seed"] in SELECTION_SEEDS else "primary_test"):
            raise ValueError("common-step CSV has invalid partition")
        for column in HEADLINE_COLUMNS:
            normalized[column] = _number(row[column], column)
        grouped[arm].append(normalized)
    if any(tuple(sorted(row["seed"] for row in values)) != tuple(range(1000, 1020)) for values in grouped.values()):
        raise ValueError("common-step CSV must have exact unique seeds for both arms")
    timesteps = {row["checkpoint_timestep"] for values in grouped.values() for row in values}
    if len(timesteps) != 1:
        raise ValueError("common-step rows must share one timestep")
    return {"timestep": int(next(iter(timesteps))), **{arm: {"primary_test": _means([row for row in rows if row["seed"] in PRIMARY_TEST_SEEDS]), "all_holdout": _means(rows)} for arm, rows in grouped.items()}}


def build_comparison_summary(root: str | Path) -> dict[str, Any]:
    """Read only canonical artifacts and return a JSON-safe, finite comparison summary."""
    base = Path(root)
    arms = {arm: _load_arm(base, arm) for arm in ARMS}
    result: dict[str, Any] = {
        "schema_version": 1,
        "primary_test_definition": {"seeds": list(PRIMARY_TEST_SEEDS), "training_runs": 1, "seed": 0},
        "selection_definition": {"seeds": list(SELECTION_SEEDS), "role": "checkpoint_selection_only"},
        "environment": _read_json(base / "environment.json"),
        "manifest": _read_json(base / "manifest.json"),
        "common_step": _common_summary(base),
    }
    for arm, data in arms.items():
        result[arm] = {"selection": _means(data["selection_rows"]), "primary_test": _means(data["primary_rows"]), "all_holdout": _means(data["all_rows"]), "runtime_metrics": data["runtime"]}
    return result


def _canonical_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_pairs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _cell(value: Any) -> str:
    return MISSING if value is None else str(value)


def _learning_plot(root: Path, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm, label in (("raw_direct", "raw-direct"), ("candidate_cnn", "candidate-CNN")):
        path = root / arm / "training_log.csv"
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows and {"timestep", "terminal_score"} <= set(rows[0]):
                axes[0].plot([_number(row["timestep"], "timestep") for row in rows], [_number(row["terminal_score"], "terminal_score") for row in rows], label=label)
        progress = root / arm / "progress_timing.csv"
        if progress.is_file():
            with progress.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows and {"recorded_training_seconds", "timestep"} <= set(rows[0]):
                axes[1].plot([_number(row["recorded_training_seconds"], "recorded_training_seconds") for row in rows], [_number(row["timestep"], "timestep") for row in rows], label=label)
    axes[0].set(title="Episode terminal score", xlabel="timestep", ylabel="score")
    axes[1].set(title="Checkpoint progress", xlabel="recorded subprocess seconds", ylabel="timestep")
    for axis in axes:
        if axis.lines: axis.legend()
        else: axis.text(.5, .5, MISSING, ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)


def _holdout_plot(summary: Mapping[str, Any], pairs: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    metric = "mean_terminal_score"
    axes[0].bar(["raw-direct", "candidate-CNN"], [summary[arm]["primary_test"][metric] for arm in ARMS])
    axes[0].set(title="Primary test terminal score (15 paired scenarios)", ylabel="score")
    axes[1].bar([str(row["seed"]) for row in pairs], [row["terminal_score_delta_cnn_minus_raw"] for row in pairs])
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(title="CNN minus raw paired terminal-score difference", xlabel="scenario seed", ylabel="delta")
    figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)


def _report_text(summary: Mapping[str, Any], pairs: list[dict]) -> str:
    env = summary["environment"]
    raw, cnn = summary["raw_direct"], summary["candidate_cnn"]
    def budget(arm: Mapping[str, Any]) -> str:
        runtime = arm["runtime_metrics"]; count = runtime.get("selection_count")
        if count is None: return MISSING
        if int(count) > 0:
            return f"validation-selected checkpoint: timestep {_cell(runtime.get('selected_checkpoint_timestep'))}, selection count {count}, tuple {_cell(runtime.get('selection_tuple'))}"
        return f"fallback_final: selection count 0 (fallback reason: periodic selection evidence unavailable)"
    pair_mean = {key: sum(row[key] for row in pairs) / len(pairs) for key in PAIR_COLUMNS[1:]}
    return f"""# Raw-direct vs Candidate-CNN 예비 결과

## 목적과 모델

이 문서는 seed 0 단일 학습 실행의 예비 결과이다. raw-direct는 grid를 읽지 않고 2772개 정규화 scalar를 직접 입력하며, candidate-CNN은 10개 작업장의 4채널 64×64 grid와 기존 구조화 pipeline을 사용한다. 두 arm은 PPO actor/critic MLP `pi=[64,64]`, `vf=[64,64]`, ReLU와 shared extractor를 사용한다.

## 통제와 실행 환경

913-block episode, deterministic ship-disjoint split, 고정 10-workspace 순서, action mask, no-rotation, reward/normalization, full context, PPO 설정(learning_rate=3e-4, n_steps=960, batch_size=64, n_epochs=10, gamma=1.0, gae_lambda=0.98), seed 0, n_envs=1, selection seed 1000..1004와 primary seed 1005..1019 분리를 고정했다. 실제 환경은 device={_cell(env.get('resolved_device'))}, GPU={_cell(env.get('gpu_name'))}, GPU UUID={_cell(env.get('gpu_uuid'))}, Torch={_cell(env.get('torch_version'))}, CUDA={_cell(env.get('cuda_version'))}이다.

## 3시간 budget 및 주 비교

각 arm의 target은 10,800 recorded training-subprocess seconds이다. raw-direct: {budget(raw)}. candidate-CNN: {budget(cnn)}. 주 성능은 checkpoint 선택에 쓰지 않은 primary_test seed 1005..1019의 15개 scenario 평균이며, 이는 15개의 독립 학습 실행이 아니라 하나의 seed 0 실행에서 짝지은 평가다. raw terminal score={raw['primary_test']['mean_terminal_score']:.6g}, CNN terminal score={cnn['primary_test']['mean_terminal_score']:.6g}; 이 수치만으로 우열 또는 통계적 유의성을 결론내리지 않는다.

## 공통 timestep과 효율

공통 timestep={summary['common_step']['timestep']}; primary terminal score는 raw={summary['common_step']['raw_direct']['primary_test']['mean_terminal_score']:.6g}, CNN={summary['common_step']['candidate_cnn']['primary_test']['mean_terminal_score']:.6g}이다. raw-direct steps/s={_cell(raw['runtime_metrics'].get('steps_per_second'))}, CNN steps/s={_cell(cnn['runtime_metrics'].get('steps_per_second'))}; parameter counts는 raw={_cell(raw['runtime_metrics'].get('parameter_counts'))}, CNN={_cell(cnn['runtime_metrics'].get('parameter_counts'))}; peak GPU memory는 raw={_cell(raw['runtime_metrics'].get('peak_cuda_memory_bytes'))}, CNN={_cell(cnn['runtime_metrics'].get('peak_cuda_memory_bytes'))}. raw arm도 동일 schema-3 환경의 shared grid construction 비용을 지불하므로 이 throughput은 grid 생성을 제거한 환경 최적화 효과를 뜻하지 않는다.

## 15개 paired scenario와 20개 보조 보기

`scenario_paired_differences.csv`는 primary_test 15개 seed의 CNN-minus-raw terminal score, dropout, mean delay days, delayed count 차이를 담는다. paired 평균은 terminal={pair_mean['terminal_score_delta_cnn_minus_raw']:.6g}, dropout={pair_mean['dropout_rate_delta_cnn_minus_raw']:.6g}, delay={pair_mean['mean_delay_days_delta_cnn_minus_raw']:.6g}, delayed count={pair_mean['delayed_count_delta_cnn_minus_raw']:.6g}이다. all_holdout 20개 평균 terminal score는 raw={raw['all_holdout']['mean_terminal_score']:.6g}, CNN={cnn['all_holdout']['mean_terminal_score']:.6g}이며 selection seed 1000..1004를 포함하는 보조 보기로서 주 일반화 성능으로 해석하지 않는다.

## 제한과 다음 단계

단일 seed 0, 순차 실행 순서, 동적 Colab 자원, parameter mismatch 때문에 통계적 유의성이나 일반적인 CNN 우월성을 주장하지 않는다. 다음 단계는 paired seed 1..4와 parameter-matched control을 추가하는 것이다. 선택적 curve 입력이 없는 경우 그래프에는 `{MISSING}`로 표시했으며 값을 0으로 만들지 않았다.
"""


def write_complete_report(root: str | Path) -> Path:
    """Write complete report artifacts; integrity marker creation belongs to Task 6."""
    base = Path(root); comparison = base / "comparison"
    summary = build_comparison_summary(base); pairs = build_paired_differences(base)
    _canonical_write_json(comparison / "summary.json", summary); _write_pairs(comparison / "scenario_paired_differences.csv", pairs)
    _learning_plot(base, comparison / "learning_curves.png"); _holdout_plot(summary, pairs, comparison / "holdout_comparison.png")
    report = comparison / "preliminary_comparison_ko.md"; text = _report_text(summary, pairs)
    if "\ufffd" in text: raise ValueError("report contains replacement character")
    report.write_text(text, encoding="utf-8", newline="\n")
    return report


def write_partial_report(root: str | Path, failure: str) -> Path:
    """Document a failed stage without fabricating report data or COMPLETE.json."""
    base = Path(root); raw_ok = (base / "raw_direct" / "runtime_metrics.json").is_file(); cnn_ok = (base / "candidate_cnn" / "runtime_metrics.json").is_file()
    text = f"# 부분 비교 보고서\n\n실패 원인: {failure}\n\n사용 가능 단계: raw-direct runtime={'있음' if raw_ok else '없음'}, candidate-CNN runtime={'있음' if cnn_ok else '없음'}.\n\n누락 단계는 재개 후 canonical 평가 CSV와 runtime metadata를 생성해야 한다. 후보 CNN 결과가 없어 우열을 결론내리지 않음. 누락 수치는 {MISSING}이며 0 또는 추정값으로 대체하지 않는다.\n"
    path = base / "comparison" / "PARTIAL_REPORT.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8", newline="\n")
    return path
