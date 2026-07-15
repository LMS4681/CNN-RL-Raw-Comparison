import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import train as train_module
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.block_generator import BlockDistribution
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import LotRegion, Workspace
from evaluation_scenarios import (
    compute_retained_choice_ratio,
    generate_scenarios,
    materialize_scenario,
    read_scenarios,
    write_scenarios,
)
from run_ablation import ABLATIONS, build_ablation_commands
from train import evaluate


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
            n_future_blocks=2,
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
            write_scenarios(path, scenarios)
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

    def test_scenario_reader_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(
                json.dumps({"schema_version": 999, "scenarios": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema"):
                read_scenarios(path)

    def test_ablation_matrix_contains_five_models_per_seed(self):
        commands = build_ablation_commands(
            "screening", [0, 1, 2], ["--data-dir", "./data"]
        )

        self.assertEqual({"A", "B", "C", "D", "E"}, set(ABLATIONS))
        self.assertEqual(15, len(commands))
        joined = [" ".join(command) for command in commands]
        self.assertTrue(
            any(
                "--extractor structured --n-future-blocks 0" in command
                for command in joined
            )
        )
        self.assertTrue(
            any(
                "--extractor candidate-cnn --n-future-blocks 4"
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

    def test_retained_choice_ratio(self):
        self.assertEqual(1.0, compute_retained_choice_ratio(0, 0))
        self.assertEqual(3.0, compute_retained_choice_ratio(0, 3))
        self.assertEqual(0.5, compute_retained_choice_ratio(4, 2))

    def test_future_choice_count_compares_the_same_future_blocks(self):
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
        env.step(0)
        after = env.future_workspace_choice_count(future_indices)

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
        self.assertEqual([1, 0], [len(ws.blocks) for ws in env._workspaces])
        self.assertEqual([0, 0], workspace_counts)
        env.close()

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
                make_plain_workspace("PE001", 30.0, 30.0),
                make_plain_workspace("PE002", 100.0, 100.0),
            ],
            seeds=[150],
            n_blocks=3,
            base_date=date(2026, 1, 5),
            spread_days=5,
        )

        rows = train_module.evaluate_fixed_scenarios(
            FirstValidModel(),
            scenarios,
            grid_size=32,
            n_future_blocks=2,
        )

        self.assertEqual(1, len(rows))
        self.assertEqual(150, rows[0]["seed"])
        self.assertIn("mean_terminal_score", rows[0])
        self.assertIn("mean_retained_choice_ratio", rows[0])

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
