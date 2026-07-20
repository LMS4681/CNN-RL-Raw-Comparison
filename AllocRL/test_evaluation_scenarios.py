import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import evaluate_baselines as baseline_module
import train as train_module
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.alloc_env import DELAY_THRESHOLD, DROPOUT_THRESHOLD
from alloc_env.block import Block, PrePlacedBlock
from alloc_env.block_generator import BlockDistribution
from alloc_env.data_split import (
    DEFAULT_SPLIT_SEED,
    sha256_file,
    split_blocks_by_ship,
)
from alloc_env.data_loader import select_workspaces_in_order
from alloc_env.strategy import BaseGridStrategy
from alloc_env.simulator import SimulationResult
from alloc_env.observation_state import (
    ObservationScales,
    build_observation_scales,
)
from alloc_env.workspace import LotRegion, Workspace
from evaluation_runner import (
    ModelActionPolicy,
    evaluate_policy,
    evaluate_scenarios,
)
from evaluation_scenarios import (
    compute_retained_choice_ratio,
    generate_scenarios,
    materialize_scenario,
    read_scenario_metadata,
    read_scenarios,
    write_scenarios,
)
from run_ablation import (
    ABLATIONS,
    _block_source_path,
    build_baseline_command,
    build_ablation_commands,
    main as run_ablation_main,
    prepare_evaluation_file,
)
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    evaluate,
    load_allocation_scenario,
    parse_workspace_codes,
)


DATA_DIR = Path(__file__).parent / "data"
BLOCK_SOURCE_FILENAME = "\ube14\ub85d\ub370\uc774\ud130.csv"
BLOCK_CSV = DATA_DIR / BLOCK_SOURCE_FILENAME


def full_source_scales() -> ObservationScales:
    return ObservationScales(
        max_length=100.0,
        max_breadth=100.0,
        max_duration=60,
        base_date=date(2025, 12, 1),
        date_span_workdays=150,
        max_workspace_area=10_000.0,
        total_workspace_area=100_000.0,
        max_workspace_length=100.0,
        max_workspace_breadth=100.0,
        dropout_threshold=7,
    )


def make_workspace(code: str = "PE001") -> Workspace:
    workspace = Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        breadth=80.0,
        length=100.0,
        name="Test workspace",
        allowable_block_patterns=["SYN-*"],
        strategy=BaseGridStrategy(step=5.0),
    )
    workspace.add_lot(
        LotRegion(
            lot_id="L1",
            origin_x=0.0,
            origin_y=0.0,
            breadth=40.0,
            length=50.0,
        )
    )
    return workspace


def make_plain_workspace(
    code: str, length: float, breadth: float
) -> Workspace:
    return Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        breadth=breadth,
        length=length,
        allowable_block_patterns=["SYN-*"],
        strategy=BaseGridStrategy(step=1.0),
    )


def make_blocks() -> list[Block]:
    return [
        Block(
            name=f"SYN-{index:05d}",
            ship_no="TEST",
            block_type="BUILD",
            length=10.0,
            breadth=10.0,
            height=2.0,
            weight=5.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 20),
        )
        for index in range(3)
    ]


class FirstValidModel:
    @staticmethod
    def predict(obs, action_masks=None, deterministic=True):
        action = int(next(i for i, valid in enumerate(action_masks) if valid))
        return action, None


class FirstValidPolicy:
    name = "first_valid"

    @staticmethod
    def select_action(env, observation):
        return int(np.flatnonzero(env.action_masks())[0])


class CountingEvaluationEnv:
    def __init__(self, delay_days=None):
        self.reset_count = 0
        self.close_count = 0
        self.delay_days = list(delay_days or [])

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.reset_count += 1
        return {"step": 0}, {}

    def action_masks(self):
        return np.array([True, False], dtype=bool)

    def step(self, action):
        if int(action) != 0:
            raise AssertionError("Policy selected a masked action")
        return (
            {"step": 1},
            2.5,
            True,
            False,
            {
                "terminal_score": 1.5,
                "raw_result": SimpleNamespace(delay_days=self.delay_days),
            },
        )

    def close(self):
        self.close_count += 1


class EvaluationScenarioTests(unittest.TestCase):
    def make_choice_env(self) -> BlockPlacementEnv:
        return BlockPlacementEnv(
            make_blocks(),
            [
                make_plain_workspace("PE001", 12.0, 12.0),
                make_plain_workspace("PE002", 100.0, 100.0),
            ],
            BaseGridStrategy(step=1.0),
            use_synthetic=False,
            grid_size=32,
            observation_scales=full_source_scales(),
        )

    def test_scenario_json_round_trip_and_materialization(self):
        scenarios = generate_scenarios(
            distribution=BlockDistribution.from_defaults(),
            workspaces=[make_workspace()],
            seeds=[100, 101],
            n_blocks=3,
            base_date=date(2026, 1, 5),
            spread_days=30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            write_scenarios(path, scenarios, {"source": "test"})
            loaded = read_scenarios(path)

        self.assertEqual(scenarios, loaded)
        blocks, workspaces = materialize_scenario(
            loaded[0], BaseGridStrategy(step=5.0)
        )
        self.assertEqual(3, len(blocks))
        self.assertEqual("PE001", workspaces[0].code)
        self.assertEqual(["SYN-*"], workspaces[0].allowable_block_patterns)
        self.assertEqual("L1", workspaces[0].lots[0].lot_id)
        self.assertTrue(all(isinstance(block.in_date, date) for block in blocks))

    def test_scenario_bundle_round_trips_provenance(self):
        blocks = make_blocks()
        workspaces = [make_workspace()]
        metadata = {
            "source": "holdout_fixed",
            "split_seed": 20260716,
            "source_sha256": "abc123",
        }
        scenarios = generate_scenarios(
            distribution=BlockDistribution.from_blocks(blocks),
            workspaces=workspaces,
            seeds=[1000],
            n_blocks=3,
            base_date=date(2026, 1, 5),
            spread_days=30,
            source_blocks=blocks,
            target_month_counts=Counter(
                (block.in_date.year, block.in_date.month)
                for block in blocks
            ),
            vary_layout=False,
            empirical_profile_probability=1.0,
            source_name="holdout_fixed",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            write_scenarios(path, scenarios, metadata)
            duplicate_path = Path(tmpdir) / "duplicate_scenarios.json"
            write_scenarios(duplicate_path, scenarios, metadata)
            self.assertEqual(scenarios, read_scenarios(path))
            self.assertEqual(metadata, read_scenario_metadata(path))
            self.assertEqual("holdout_fixed", scenarios[0]["source"])
            self.assertEqual(path.read_bytes(), duplicate_path.read_bytes())

    def test_block_source_path_uses_exact_filename_despite_extra_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            copied_data_dir = Path(tmpdir) / "data"
            shutil.copytree(DATA_DIR, copied_data_dir)
            (copied_data_dir / "00-distractor.csv").write_bytes(
                b"STAGE,distractor\n"
            )

            self.assertEqual(
                copied_data_dir / BLOCK_SOURCE_FILENAME,
                _block_source_path(copied_data_dir),
            )

    def test_block_source_path_rejects_missing_or_invalid_exact_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "source file"):
                _block_source_path(data_dir)

            (data_dir / BLOCK_SOURCE_FILENAME).write_bytes(b"invalid\n")
            with self.assertRaisesRegex(ValueError, "invalid"):
                _block_source_path(data_dir)

    def test_prepare_evaluation_file_uses_holdout_templates_and_full_profile(
        self,
    ):
        blocks, _ = load_allocation_scenario(
            DATA_DIR,
            BaseGridStrategy(step=5.0),
            parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
        )
        split = split_blocks_by_ship(blocks, BLOCK_CSV)
        target_month_counts = Counter(
            (block.in_date.year, block.in_date.month) for block in blocks
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "fixed_eval_scenarios.json"
            prepare_evaluation_file(DATA_DIR, scenario_path)
            scenarios = read_scenarios(scenario_path)
            metadata = read_scenario_metadata(scenario_path)
            manifest_path = Path(tmpdir) / "data_split_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(list(range(1000, 1020)), [
            scenario["seed"] for scenario in scenarios
        ])
        self.assertEqual(20, len(scenarios))
        self.assertEqual(manifest, metadata)
        self.assertEqual("holdout_fixed", metadata["source"])
        self.assertEqual(DEFAULT_SPLIT_SEED, metadata["split_seed"])
        self.assertEqual(
            sha256_file(BLOCK_CSV), metadata["source_sha256"]
        )
        self.assertEqual(913, metadata["source_row_count"])
        self.assertEqual(240, metadata["holdout_row_count"])
        self.assertEqual(40, metadata["source_ship_count"])
        self.assertEqual(29, metadata["training_ship_count"])
        self.assertEqual(11, metadata["holdout_ship_count"])
        self.assertEqual(
            split.manifest["source_month_counts"],
            metadata["target_month_counts"],
        )
        self.assertEqual(
            "holdout_ship_split",
            metadata["provenance"]["template_source"],
        )
        self.assertFalse(metadata["provenance"]["vary_layout"])
        self.assertEqual(
            1.0,
            metadata["provenance"]["empirical_profile_probability"],
        )
        holdout_ships = set(split.manifest["holdout_ship_nos"])
        for scenario in scenarios:
            self.assertEqual("holdout_fixed", scenario["source"])
            self.assertEqual(913, len(scenario["blocks"]))
            self.assertEqual(10, len(scenario["workspaces"]))
            self.assertTrue(
                all(not workspace["pre_placements"]
                    for workspace in scenario["workspaces"])
            )
            self.assertEqual(
                target_month_counts,
                Counter(
                    (date.fromisoformat(block["in_date"]).year,
                     date.fromisoformat(block["in_date"]).month)
                    for block in scenario["blocks"]
                ),
            )
            self.assertTrue(
                {block["ship_no"] for block in scenario["blocks"]}
                .issubset(holdout_ships)
            )

    def test_prepared_artifact_bytes_match_committed_bundle_and_manifest(self):
        committed_scenarios = (
            DATA_DIR / "fixed_eval_scenarios.json"
        ).read_bytes()
        committed_manifest = (
            DATA_DIR / "data_split_manifest.json"
        ).read_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_path = Path(tmpdir) / "fixed_eval_scenarios.json"
            prepare_evaluation_file(DATA_DIR, prepared_path)
            prepared_scenarios = prepared_path.read_bytes()
            prepared_manifest = prepared_path.with_name(
                "data_split_manifest.json"
            ).read_bytes()

        self.assertEqual(committed_scenarios, prepared_scenarios)
        self.assertEqual(
            hashlib.sha256(committed_scenarios).hexdigest(),
            hashlib.sha256(prepared_scenarios).hexdigest(),
        )
        self.assertEqual(committed_manifest, prepared_manifest)

    def test_source_scenarios_keep_workspace_geometry_and_remove_obstacles(self):
        workspace = make_workspace()
        workspace.add_pre_placement(PrePlacedBlock(
            label="old",
            pos_x=10.0,
            pos_y=10.0,
            length=5.0,
            breadth=5.0,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        ))
        source = make_blocks()

        scenarios = generate_scenarios(
            distribution=BlockDistribution.from_blocks(source),
            workspaces=[workspace],
            seeds=[100],
            n_blocks=3,
            base_date=date(2026, 1, 5),
            spread_days=30,
            source_blocks=source,
            vary_layout=False,
            empirical_profile_probability=1.0,
        )

        record = scenarios[0]["workspaces"][0]
        self.assertEqual(100.0, record["length"])
        self.assertEqual(80.0, record["breadth"])
        self.assertEqual([], record["pre_placements"])
        self.assertEqual(3, len(scenarios[0]["blocks"]))

    def test_scenario_reader_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(
                json.dumps({"schema_version": 999, "scenarios": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema"):
                read_scenarios(path)

    def test_scenario_reader_rejects_legacy_five_workspace_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_scenarios.json"
            path.write_text(
                json.dumps({"schema_version": 1, "scenarios": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema"):
                read_scenarios(path)

    def test_scenario_reader_rejects_non_object_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(json.dumps([]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "object"):
                read_scenarios(path)

    def test_scenario_reader_rejects_scenario_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(
                json.dumps({
                    "schema_version": 3,
                    "metadata": {},
                    "scenarios": [{
                        "seed": 1000,
                        "source": "holdout_fixed",
                        "blocks": [],
                    }],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "seed, source"):
                read_scenarios(path)

    def test_ablation_matrix_contains_five_models_per_seed(self):
        commands = build_ablation_commands(
            "screening", [0, 1, 2], ["--data-dir", "./data"]
        )

        self.assertEqual(
            {
                "A": ("structured", "current"),
                "B": ("structured", "full"),
                "C": ("fixed-grid", "full"),
                "D": ("candidate-cnn", "current"),
                "E": ("candidate-cnn", "full"),
            },
            ABLATIONS,
        )
        self.assertEqual(15, len(commands))
        joined = [" ".join(command) for command in commands]
        self.assertTrue(
            any(
                "--extractor structured --state-context current" in command
                for command in joined
            )
        )
        self.assertTrue(
            any(
                "--extractor candidate-cnn --state-context full"
                in command
                for command in joined
            )
        )
        self.assertTrue(
            all(
                "--eval-scenarios ./data/fixed_eval_scenarios.json"
                in command
                for command in joined
            )
        )
        self.assertTrue(
            all("--timesteps 20000" in command for command in joined)
        )

    def test_final_ablation_uses_full_budget(self):
        commands = build_ablation_commands("final", [7], [])

        self.assertEqual(5, len(commands))
        self.assertTrue(
            all("--timesteps 100000" in " ".join(c) for c in commands)
        )

    def test_baseline_command_uses_the_fixed_holdout_bundle(self):
        output = "./output_ablation/baselines/evaluation_scenarios.csv"

        command = build_baseline_command(
            scenario_path="./data/fixed_eval_scenarios.json",
            output_path=output,
        )

        joined = subprocess.list2cmdline(command)
        self.assertIn("evaluate_baselines.py", joined)
        self.assertIn("fixed_eval_scenarios.json", joined)
        self.assertIn(output, joined)

    def test_evaluate_baselines_dispatches_the_list_command(self):
        expected = build_baseline_command(
            "./data/fixed_eval_scenarios.json",
            "./output_ablation/baselines/evaluation_scenarios.csv",
        )

        with (
            patch.object(
                sys, "argv", ["run_ablation.py", "--evaluate-baselines"]
            ),
            patch("run_ablation.subprocess.run") as run,
        ):
            run_ablation_main()

        run.assert_called_once_with(expected, check=True)

    def test_baseline_main_builds_scales_from_full_source_once(self):
        scenarios = [
            {
                "seed": 1000,
                "source": "holdout_fixed",
                "blocks": [],
                "workspaces": [],
            }
        ]
        full_blocks = [object()] * 913
        workspaces = [object()] * 10
        scales = full_source_scales()
        row = {
            "source": "holdout_fixed20",
            "policy": "stub",
            "seed": 1000,
            "mean_terminal_score": 0.0,
            "mean_dropout_rate": 0.0,
            "mean_delay_days": 0.0,
        }

        with (
            patch.object(
                sys,
                "argv",
                ["evaluate_baselines.py", "--data-dir", "./data"],
            ),
            patch.object(
                baseline_module, "read_scenarios", return_value=scenarios
            ),
            patch.object(
                baseline_module,
                "load_allocation_scenario",
                return_value=(full_blocks, workspaces),
                create=True,
            ),
            patch.object(
                baseline_module,
                "build_observation_scales",
                return_value=scales,
                create=True,
            ) as build_scales,
            patch.object(
                baseline_module,
                "evaluate_scenarios",
                return_value=[row],
            ) as evaluate,
            patch.object(baseline_module, "write_evaluation_metrics"),
        ):
            baseline_module.main()

        build_scales.assert_called_once()
        self.assertIs(full_blocks, build_scales.call_args.args[0])
        self.assertIs(workspaces, build_scales.call_args.args[1])
        self.assertEqual(2, evaluate.call_count)
        for call in evaluate.call_args_list:
            self.assertIs(scales, call.kwargs["observation_scales"])
            self.assertEqual("full", call.kwargs["state_context_mode"])

    def test_retained_choice_ratio(self):
        self.assertEqual(1.0, compute_retained_choice_ratio(0, 0))
        self.assertEqual(3.0, compute_retained_choice_ratio(0, 3))
        self.assertEqual(0.5, compute_retained_choice_ratio(4, 2))

    def test_model_policy_passes_action_mask_and_requests_determinism(self):
        calls = []

        class RecordingModel:
            @staticmethod
            def predict(observation, **kwargs):
                calls.append((observation, kwargs))
                return np.array(0), None

        env = CountingEvaluationEnv()
        observation = {"step": 0}

        action = ModelActionPolicy(RecordingModel()).select_action(
            env, observation
        )

        self.assertEqual(0, action)
        self.assertEqual(observation, calls[0][0])
        np.testing.assert_array_equal(
            np.array([True, False], dtype=bool),
            calls[0][1]["action_masks"],
        )
        self.assertIs(True, calls[0][1]["deterministic"])

    def test_shared_evaluation_preserves_metric_semantics(self):
        env = CountingEvaluationEnv([
            SimulationResult.DROPOUT,
            0,
            DELAY_THRESHOLD + 1,
        ])

        metrics = evaluate_policy(FirstValidPolicy(), env, episodes=1)

        self.assertEqual(1, env.reset_count)
        self.assertEqual(2.5, metrics["mean_reward"])
        self.assertEqual(1.5, metrics["mean_terminal_score"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["mean_dropout_rate"])
        self.assertEqual(1.5, metrics["mean_delay_days"])
        self.assertEqual(1.0, metrics["mean_delayed_count"])
        self.assertEqual(1.0, metrics["mean_retained_choice_ratio"])

    def test_shared_evaluation_uses_immediate_choice_preview(self):
        env = CountingEvaluationEnv()
        env.future_workspace_choice_indices = lambda: [1]
        env.future_workspace_choice_count = lambda indices: 4
        env.future_workspace_choice_count_after_action = (
            lambda action, indices: 1
        )

        metrics = evaluate_policy(FirstValidPolicy(), env)

        self.assertEqual(0.25, metrics["mean_retained_choice_ratio"])

    def test_shared_evaluation_rejects_non_positive_episode_count(self):
        env = CountingEvaluationEnv()

        with self.assertRaisesRegex(ValueError, "at least 1"):
            evaluate_policy(FirstValidPolicy(), env, episodes=0)

        self.assertEqual(0, env.reset_count)

    def test_cli_accepts_deprecated_n_eval_and_evaluates_csv_once(self):
        env = CountingEvaluationEnv()
        parsed_n_eval = []
        result_rows = []

        def run_from_parsed_args(args):
            parsed_n_eval.append(args.n_eval)
            with self.assertWarnsRegex(FutureWarning, "--n-eval.*ignored"):
                result_rows.append(
                    train_module.evaluate_original_csv_row(
                        FirstValidModel(), env, n_eval=args.n_eval
                    )
                )

        with (
            patch("sys.argv", ["train.py", "--n-eval", "5"]),
            patch.object(
                train_module, "train", side_effect=run_from_parsed_args
            ) as train_entry,
        ):
            train_module.main()

        self.assertEqual([5], parsed_n_eval)
        self.assertEqual(1, train_entry.call_count)
        self.assertEqual(1, env.reset_count)
        self.assertEqual([
            {
                "source": "original_csv",
                "policy": "model",
                "mean_reward": 2.5,
                "mean_terminal_score": 1.5,
                "mean_dropout_rate": 0.0,
                "mean_delay_days": 0.0,
                "mean_delayed_count": 0.0,
                "mean_retained_choice_ratio": 1.0,
            }
        ], result_rows)

    def test_future_choice_count_previews_immediate_post_action_state(self):
        env = self.make_choice_env()
        self.assertEqual(0, env.future_workspace_choice_count())
        env.reset(seed=0)
        future_indices = env.future_workspace_choice_indices()
        block_state = [
            (block.length, block.breadth, block.angle)
            for block in env._blocks
        ]
        workspace_counts = [len(ws.blocks) for ws in env._workspaces]

        before = env.future_workspace_choice_count(future_indices)
        after = env.future_workspace_choice_count_after_action(
            0, future_indices
        )

        self.assertEqual(4, before)
        self.assertEqual(2, after)
        self.assertEqual(0.5, compute_retained_choice_ratio(before, after))
        self.assertEqual(
            block_state[1:],
            [
                (block.length, block.breadth, block.angle)
                for block in env._blocks[1:]
            ],
        )
        self.assertEqual([0, 0], workspace_counts)
        self.assertEqual([0, 0], [len(ws.blocks) for ws in env._workspaces])
        env.close()

    def test_evaluate_uses_immediate_post_action_choice_preview(self):
        env = self.make_choice_env()
        env.future_workspace_choice_indices = lambda: [1]
        env.future_workspace_choice_count = lambda block_indices=None: 4
        env.future_workspace_choice_count_after_action = (
            lambda action, block_indices=None: 1
        )

        metrics = evaluate(
            FirstValidModel(), env, n_eval=1, return_metrics=True
        )
        env.close()

        self.assertEqual(0.25, metrics["mean_retained_choice_ratio"])

    def test_evaluate_can_return_quality_and_retained_choice_metrics(self):
        env = self.make_choice_env()
        metrics = evaluate(
            FirstValidModel(), env, n_eval=1, return_metrics=True
        )
        env.close()

        self.assertAlmostEqual(
            metrics["mean_reward"], metrics["mean_terminal_score"]
        )
        self.assertGreaterEqual(metrics["mean_dropout_rate"], 0.0)
        self.assertLessEqual(metrics["mean_dropout_rate"], 1.0)
        self.assertGreaterEqual(metrics["mean_delay_days"], 0.0)
        self.assertGreaterEqual(metrics["mean_delayed_count"], 0.0)
        self.assertGreaterEqual(metrics["mean_retained_choice_ratio"], 0.0)

    def test_requested_scenario_file_must_exist_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.json"

            with self.assertRaisesRegex(
                FileNotFoundError, "--prepare-eval-scenarios"
            ):
                train_module.load_requested_evaluation_scenarios(missing)

    def test_fixed_scenario_evaluation_returns_seeded_metric_rows(self):
        scenarios = generate_scenarios(
            distribution=BlockDistribution.from_defaults(),
            workspaces=[
                make_plain_workspace(f"PE{index:03d}", 100.0, 100.0)
                for index in range(1, 11)
            ],
            seeds=[150],
            n_blocks=3,
            base_date=date(2026, 1, 5),
            spread_days=5,
            vary_layout=False,
        )

        rows = train_module.evaluate_fixed_scenarios(
            FirstValidModel(),
            scenarios,
            observation_scales=full_source_scales(),
            state_context_mode="full",
        )

        self.assertEqual(1, len(rows))
        self.assertEqual(150, rows[0]["seed"])
        self.assertIn("mean_terminal_score", rows[0])
        self.assertIn("mean_retained_choice_ratio", rows[0])

    def test_selected_holdout_report_loads_exact_best_model_and_tags_all_20(
        self,
    ):
        scenarios = [{"seed": seed} for seed in range(1000, 1020)]
        evaluated = {}

        class CloseCountingEnv:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        training_env = CloseCountingEnv()

        class SelectedModel:
            def __init__(self, env):
                self.env = env

            def get_env(self):
                return self.env

        class FakeMaskablePPO:
            loaded = None

            @classmethod
            def load(cls, path, **kwargs):
                cls.loaded = (path, kwargs)
                return SelectedModel(kwargs["env"])

        def evaluate_fn(policy_factory, scenario_records):
            evaluated["policy"] = policy_factory(1000)
            evaluated["scenarios"] = scenario_records
            return [
                {"seed": item["seed"]}
                for item in scenario_records
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = Path(tmpdir) / "best_model.sb3"
            with zipfile.ZipFile(best_path, "w") as archive:
                archive.writestr("probe", "selected")

            rows = train_module.evaluate_selected_holdout_report(
                FakeMaskablePPO,
                output_dir=tmpdir,
                training_env=training_env,
                scenario_records=scenarios,
                device="cpu",
                evaluate_fn=evaluate_fn,
            )

        loaded_path, load_kwargs = FakeMaskablePPO.loaded
        self.assertEqual(str(best_path.resolve()), loaded_path)
        self.assertIs(training_env, load_kwargs["env"])
        self.assertEqual("cpu", load_kwargs["device"])
        self.assertEqual(list(range(1000, 1020)), [row["seed"] for row in rows])
        self.assertTrue(all(row["checkpoint"] == "best_model" for row in rows))
        self.assertIsInstance(evaluated["policy"].model, SelectedModel)
        self.assertEqual(1, training_env.close_count)
        self.assertIs(scenarios, evaluated["scenarios"])

    def test_selected_holdout_report_requires_best_model(self):
        model_env = CountingEvaluationEnv()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "best_model.sb3"):
                train_module.evaluate_selected_holdout_report(
                    object(),
                    output_dir=tmpdir,
                    training_env=model_env,
                    scenario_records=[
                        {"seed": seed} for seed in range(1000, 1020)
                    ],
                    device="cpu",
                    evaluate_fn=lambda policy_factory, scenarios: [],
                )

        self.assertEqual(1, model_env.close_count)

    def test_all_real_fixed_scenarios_obey_and_reset_with_full_source_scales(
        self,
    ):
        strategy = BaseGridStrategy(step=5.0)
        workspace_codes = parse_workspace_codes(
            DEFAULT_ACTIVE_WORKSPACE_CODES
        )
        full_blocks, source_workspaces = load_allocation_scenario(
            DATA_DIR, strategy, workspace_codes
        )
        scales = build_observation_scales(
            full_blocks, source_workspaces, DROPOUT_THRESHOLD
        )
        scenario_path = DATA_DIR / "fixed_eval_scenarios.json"
        scenarios = read_scenarios(scenario_path)
        source_split = split_blocks_by_ship(full_blocks, BLOCK_CSV)
        expected_month_counts = Counter(
            (block.in_date.year, block.in_date.month)
            for block in full_blocks
        )
        holdout_ships = set(source_split.manifest["holdout_ship_nos"])

        self.assertEqual(date(2025, 12, 4), scales.base_date)
        self.assertEqual(list(range(1000, 1020)), [
            scenario["seed"] for scenario in scenarios
        ])
        for scenario in scenarios:
            self.assertEqual("holdout_fixed", scenario["source"])
            self.assertEqual(913, len(scenario["blocks"]))
            self.assertEqual(10, len(scenario["workspaces"]))
            self.assertEqual(
                workspace_codes,
                [workspace["code"] for workspace in scenario["workspaces"]],
            )
            self.assertGreaterEqual(
                min(
                    date.fromisoformat(block["in_date"])
                    for block in scenario["blocks"]
                ),
                scales.base_date,
            )
            self.assertEqual(
                expected_month_counts,
                Counter(
                    (
                        date.fromisoformat(block["in_date"]).year,
                        date.fromisoformat(block["in_date"]).month,
                    )
                    for block in scenario["blocks"]
                ),
            )
            self.assertTrue(
                {block["ship_no"] for block in scenario["blocks"]}
                .issubset(holdout_ships)
            )
            self.assertTrue(all(
                not workspace["pre_placements"]
                for workspace in scenario["workspaces"]
            ))

            scenario_strategy = BaseGridStrategy(step=5.0)
            blocks, workspaces = materialize_scenario(
                scenario, scenario_strategy
            )
            ordered = select_workspaces_in_order(workspaces, workspace_codes)
            env = train_module.create_evaluation_env(
                blocks,
                ordered,
                scenario_strategy,
                observation_scales=scales,
                state_context_mode="full",
                seed=int(scenario["seed"]),
            )
            try:
                observation, _ = env.reset()
                self.assertIs(scales, env.unwrapped._observation_scales)
                self.assertTrue(env.observation_space.contains(observation))
            finally:
                env.close()

    def test_shared_scenarios_create_and_close_env_and_policy_per_seed(self):
        scenarios = [
            {"seed": 11, "source": "holdout_fixed", "blocks": [], "workspaces": []},
            {"seed": 12, "source": "holdout_fixed", "blocks": [], "workspaces": []},
        ]
        envs = [CountingEvaluationEnv(), CountingEvaluationEnv()]
        policies = []
        scales = full_source_scales()

        def policy_factory(seed):
            policy = FirstValidPolicy()
            policy.name = f"first_valid_{seed}"
            policies.append(policy)
            return policy

        with (
            patch("evaluation_runner.materialize_scenario", return_value=([], [])),
            patch("evaluation_runner.select_workspaces_in_order", return_value=[]),
            patch("train.create_evaluation_env", side_effect=envs) as create_env,
        ):
            rows = evaluate_scenarios(
                policy_factory,
                scenarios,
                workspace_codes=["PE001"],
                observation_scales=scales,
                state_context_mode="current",
            )

        self.assertEqual([11, 12], [row["seed"] for row in rows])
        self.assertEqual(
            ["first_valid_11", "first_valid_12"],
            [row["policy"] for row in rows],
        )
        self.assertEqual(
            ["holdout_fixed20", "holdout_fixed20"],
            [row["source"] for row in rows],
        )
        self.assertEqual(2, create_env.call_count)
        for call in create_env.call_args_list:
            self.assertEqual(64, call.kwargs["grid_size"])
            self.assertIs(scales, call.kwargs["observation_scales"])
            self.assertEqual("current", call.kwargs["state_context_mode"])
        self.assertIsNot(policies[0], policies[1])
        self.assertEqual([1, 1], [env.reset_count for env in envs])
        self.assertEqual([1, 1], [env.close_count for env in envs])

    def test_shared_scenario_closes_env_when_policy_creation_fails(self):
        scenario = {
            "seed": 11,
            "source": "holdout_fixed",
            "blocks": [],
            "workspaces": [],
        }
        env = CountingEvaluationEnv()
        scales = full_source_scales()

        with (
            patch("evaluation_runner.materialize_scenario", return_value=([], [])),
            patch("evaluation_runner.select_workspaces_in_order", return_value=[]),
            patch("train.create_evaluation_env", return_value=env),
            self.assertRaisesRegex(RuntimeError, "factory failed"),
        ):
            evaluate_scenarios(
                lambda seed: (_ for _ in ()).throw(
                    RuntimeError("factory failed")
                ),
                [scenario],
                workspace_codes=None,
                observation_scales=scales,
            )

        self.assertEqual(1, env.close_count)

    def test_evaluation_metric_rows_are_written_as_csv(self):
        rows = [
            {
                "seed": 100,
                "mean_reward": 0.5,
                "mean_terminal_score": 0.5,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evaluation.csv"
            train_module.write_evaluation_metrics(path, rows)
            with path.open(encoding="utf-8", newline="") as file:
                loaded = list(csv.DictReader(file))

        self.assertEqual("100", loaded[0]["seed"])
        self.assertEqual("0.5", loaded[0]["mean_terminal_score"])


if __name__ == "__main__":
    unittest.main()
