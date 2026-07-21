import unittest

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)


N_WORKSPACES = 10
EXTRACTOR_CLASSES = (
    StructuredExtractor,
    FixedGridExtractor,
    CandidateCnnExtractor,
)


def observation_space(grid_size: int = 64) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (N_WORKSPACES, 8), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 6), np.float32),
        "pending_blocks": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 32, 7), np.float32
        ),
        "pending_mask": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 32), np.float32
        ),
        "pending_summary": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4), np.float32
        ),
    })


def observation(
    batch_size: int = 2,
    grid_size: int = 64,
) -> dict[str, torch.Tensor]:
    return {
        "block": torch.rand(batch_size, 8),
        "grids": torch.rand(
            batch_size, N_WORKSPACES, 4, grid_size, grid_size
        ),
        "ws_meta": torch.rand(batch_size, N_WORKSPACES, 8),
        "future_blocks": torch.rand(batch_size, 16, 6),
        "future_mask": torch.ones(batch_size, 16),
        "future_demand": torch.rand(batch_size, 3, 6),
        "pending_blocks": torch.rand(
            batch_size, N_WORKSPACES, 32, 7
        ),
        "pending_mask": torch.ones(
            batch_size, N_WORKSPACES, 32
        ),
        "pending_summary": torch.rand(
            batch_size, N_WORKSPACES, 4
        ),
    }


def clone_observation(
    observations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in observations.items()}


def zero_observation(
    batch_size: int = 1,
    grid_size: int = 16,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(value)
        for key, value in observation(batch_size, grid_size).items()
    }


def initialize_position_sensitive_linears(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for layer in module.modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            columns = torch.arange(
                1,
                layer.in_features + 1,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            ) / layer.in_features
            rows = torch.linspace(
                0.5,
                1.0,
                layer.out_features,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
            layer.weight.copy_(rows.unsqueeze(1) * columns.unsqueeze(0))
            layer.bias.fill_(0.1)


def capture_structured_inputs(
    extractor: StructuredExtractor,
    observations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    captured = {}
    handles = []
    for name in ("current", "future", "demand", "pending"):
        first_linear = getattr(extractor.structured_encoder, name)[0]

        def capture(_module, inputs, key=name):
            captured[key] = inputs[0].detach().clone()

        handles.append(first_linear.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            extractor.structured_encoder(observations)
    finally:
        for handle in handles:
            handle.remove()
    return captured


@pytest.mark.parametrize("extractor_class", EXTRACTOR_CLASSES)
@pytest.mark.parametrize(
    ("malformed_case", "message"),
    (
        ("missing", "nine keys.*missing grids"),
        ("extra", "nine keys.*unexpected extra"),
        ("workspace_count", "grids.*10"),
        ("legacy_ws_meta", "ws_meta.*10, 8"),
        ("structured_trailing", "pending_blocks.*7"),
        ("grid_rank", "grids.*shape"),
        ("grid_channels", "grids.*4"),
    ),
)
def test_every_extractor_rejects_malformed_schema_with_value_error(
    extractor_class,
    malformed_case: str,
    message: str,
):
    space = observation_space(grid_size=16)
    if malformed_case == "missing":
        del space.spaces["grids"]
    elif malformed_case == "extra":
        space.spaces["extra"] = gym.spaces.Box(0, 1, (1,), np.float32)
    elif malformed_case == "workspace_count":
        space.spaces["grids"] = gym.spaces.Box(
            0, 1, (9, 4, 16, 16), np.float32
        )
    elif malformed_case == "legacy_ws_meta":
        space.spaces["ws_meta"] = gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4), np.float32
        )
    elif malformed_case == "structured_trailing":
        space.spaces["pending_blocks"] = gym.spaces.Box(
            0, 1, (N_WORKSPACES, 32, 8), np.float32
        )
    elif malformed_case == "grid_rank":
        space.spaces["grids"] = gym.spaces.Box(
            0, 1, (N_WORKSPACES,), np.float32
        )
    elif malformed_case == "grid_channels":
        space.spaces["grids"] = gym.spaces.Box(
            0, 1, (N_WORKSPACES, 3, 16, 16), np.float32
        )
    else:
        raise AssertionError(f"unhandled malformed case {malformed_case}")

    with pytest.raises(ValueError, match=message):
        extractor_class(space)


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.space = observation_space()

    def test_all_extractors_return_batch_by_features_dim_and_finite_values(self):
        observations = observation()
        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            with self.subTest(extractor=extractor_class.__name__):
                extractor = extractor_class(self.space, features_dim=256)
                features = extractor(observations)
                self.assertEqual((2, 256), tuple(features.shape))
                self.assertTrue(torch.isfinite(features).all())

    def test_structured_encoder_uses_exact_network_dimensions(self):
        extractor = StructuredExtractor(self.space, features_dim=64)
        encoder = extractor.structured_encoder

        expected = {
            "current": [(8, 64), (64, 32)],
            "future": [(112, 128), (128, 64)],
            "demand": [(18, 64), (64, 32)],
            "pending": [(256, 128), (128, 64)],
        }
        for name, dimensions in expected.items():
            with self.subTest(network=name):
                linears = [
                    module
                    for module in getattr(encoder, name).modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(
                    dimensions,
                    [
                        (layer.in_features, layer.out_features)
                        for layer in linears
                    ],
                )

        for extractor_class, workspace_input_dim in (
            (StructuredExtractor, 204),
            (FixedGridExtractor, 460),
            (CandidateCnnExtractor, 332),
        ):
            with self.subTest(fusion=extractor_class.__name__):
                extractor = extractor_class(self.space, features_dim=64)
                fusion_linears = [
                    module
                    for module in extractor.workspace_fusion.modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(
                    [(workspace_input_dim, 128), (128, 64)],
                    [
                        (layer.in_features, layer.out_features)
                        for layer in fusion_linears
                    ],
                )
                self.assertEqual(
                    (640, 64),
                    (
                        extractor.global_fusion[0].in_features,
                        extractor.global_fusion[0].out_features,
                    ),
                )

    def test_structured_preprocessing_maps_values_masks_and_order_exactly(self):
        space = observation_space(grid_size=16)
        observations = zero_observation()
        current_values = torch.linspace(0.1, 0.8, 8)
        demand_values = torch.linspace(0.05, 0.9, 18)
        future_first = torch.linspace(0.1, 0.6, 6)
        future_third = torch.linspace(0.2, 0.7, 6)
        pending_first = torch.linspace(0.1, 0.7, 7)
        pending_second_workspace = torch.linspace(0.2, 0.8, 7)

        observations["block"][0] = current_values
        observations["future_demand"][0] = demand_values.reshape(3, 6)

        observations["future_blocks"][0, 0] = future_first
        observations["future_mask"][0, 0] = 1.0
        observations["future_blocks"][0, 1] = 0.9
        observations["future_blocks"][0, 2] = future_third
        observations["future_mask"][0, 2] = 1.0

        observations["pending_blocks"][0, 0, 0] = pending_first
        observations["pending_mask"][0, 0, 0] = 1.0
        observations["pending_blocks"][0, 0, 1] = 0.9
        observations["pending_blocks"][0, 1, 2] = (
            pending_second_workspace
        )
        observations["pending_mask"][0, 1, 2] = 1.0

        extractor = StructuredExtractor(space, features_dim=64)
        captured = capture_structured_inputs(extractor, observations)

        torch.testing.assert_close(captured["current"], observations["block"])
        torch.testing.assert_close(
            captured["demand"],
            demand_values.reshape(1, 18),
        )

        expected_future = torch.zeros(1, 112)
        expected_future[0, :6] = future_first
        expected_future[0, 6] = 1.0
        expected_future[0, 14:20] = future_third
        expected_future[0, 20] = 1.0
        torch.testing.assert_close(captured["future"], expected_future)

        expected_pending = torch.zeros(1, N_WORKSPACES, 256)
        expected_pending[0, 0, :7] = pending_first
        expected_pending[0, 0, 7] = 1.0
        expected_pending[0, 1, 16:23] = pending_second_workspace
        expected_pending[0, 1, 23] = 1.0
        torch.testing.assert_close(captured["pending"], expected_pending)

    def test_zero_features_with_one_mask_bit_remain_observable(self):
        extractor = StructuredExtractor(
            observation_space(grid_size=16), features_dim=64
        )
        observations = zero_observation()
        observations["future_mask"][0, 3] = 1.0
        observations["pending_mask"][0, 4, 5] = 1.0

        captured = capture_structured_inputs(extractor, observations)

        expected_future = torch.zeros(1, 112)
        expected_future[0, 3 * 7 + 6] = 1.0
        expected_pending = torch.zeros(1, N_WORKSPACES, 256)
        expected_pending[0, 4, 5 * 8 + 7] = 1.0
        torch.testing.assert_close(captured["future"], expected_future)
        torch.testing.assert_close(captured["pending"], expected_pending)

    def test_fusion_positions_match_structured_meta_and_summary_sources(self):
        observations = zero_observation()
        observations["block"][0] = torch.linspace(0.1, 0.8, 8)
        observations["future_mask"][0, 0] = 1.0
        observations["future_demand"][0, 0, 0] = 0.25
        observations["pending_mask"][0, 2, 3] = 1.0
        observations["pending_summary"][0] = torch.linspace(
            0.0, 0.975, 40
        ).reshape(10, 4)
        observations["ws_meta"][0] = torch.linspace(
            0.0125, 1.0, 80
        ).reshape(10, 8)

        for extractor_class in EXTRACTOR_CLASSES:
            extractor = extractor_class(
                observation_space(grid_size=16), features_dim=64
            ).eval()
            captured = {}

            def capture_structured(_module, _inputs, output):
                captured["structured"] = tuple(
                    value.detach().clone() for value in output
                )

            def capture_fusion(_module, inputs):
                captured["fusion"] = inputs[0].detach().clone()

            def capture_workspace_output(_module, _inputs, output):
                captured["workspace_output"] = output.detach().clone()

            def capture_global(_module, inputs):
                captured["global"] = inputs[0].detach().clone()

            handles = (
                extractor.structured_encoder.register_forward_hook(
                    capture_structured
                ),
                extractor.workspace_fusion[0].register_forward_pre_hook(
                    capture_fusion
                ),
                extractor.workspace_fusion.register_forward_hook(
                    capture_workspace_output
                ),
                extractor.global_fusion[0].register_forward_pre_hook(
                    capture_global
                ),
            )
            try:
                with torch.no_grad():
                    extractor(observations)
            finally:
                for handle in handles:
                    handle.remove()

            current, future, demand, pending = captured["structured"]
            fusion = captured["fusion"]
            with self.subTest(extractor=extractor_class.__name__):
                torch.testing.assert_close(
                    fusion[:, :, 0:32],
                    current.unsqueeze(1).expand(-1, 10, -1),
                )
                torch.testing.assert_close(
                    fusion[:, :, 32:96],
                    future.unsqueeze(1).expand(-1, 10, -1),
                )
                torch.testing.assert_close(
                    fusion[:, :, 96:128],
                    demand.unsqueeze(1).expand(-1, 10, -1),
                )
                torch.testing.assert_close(fusion[:, :, 128:192], pending)
                torch.testing.assert_close(
                    fusion[:, :, 192:196], observations["pending_summary"]
                )
                torch.testing.assert_close(
                    fusion[:, :, 196:204], observations["ws_meta"]
                )
                torch.testing.assert_close(
                    captured["global"],
                    captured["workspace_output"].flatten(1),
                )

    def test_deterministic_end_to_end_structured_sensitivity(self):
        original = zero_observation()
        original["future_mask"].fill_(1.0)
        original["pending_mask"].fill_(1.0)

        mutations = {
            "block": lambda changed: changed["block"].fill_(1.0),
            "future_blocks": lambda changed: changed[
                "future_blocks"
            ].fill_(1.0),
            "future_mask": lambda changed: changed[
                "future_mask"
            ].zero_(),
            "future_demand": lambda changed: changed[
                "future_demand"
            ].fill_(1.0),
            "pending_blocks": lambda changed: changed[
                "pending_blocks"
            ].fill_(1.0),
            "pending_mask": lambda changed: changed[
                "pending_mask"
            ].zero_(),
            "pending_summary": lambda changed: changed[
                "pending_summary"
            ].fill_(1.0),
            "ws_meta": lambda changed: changed["ws_meta"].fill_(1.0),
        }

        for extractor_class in EXTRACTOR_CLASSES:
            extractor = extractor_class(
                observation_space(grid_size=16), features_dim=64
            ).eval()
            initialize_position_sensitive_linears(extractor)
            with torch.no_grad():
                baseline = extractor(original)
                for key, mutate in mutations.items():
                    changed = clone_observation(original)
                    mutate(changed)
                    delta = (extractor(changed) - baseline).abs().max().item()
                    with self.subTest(
                        extractor=extractor_class.__name__, key=key
                    ):
                        self.assertGreater(delta, 1e-6)

    def test_masked_future_and_pending_padding_values_are_invariant(self):
        space = observation_space(grid_size=16)
        original = observation(batch_size=1, grid_size=16)
        original["future_mask"][:, 2:] = 0.0
        original["pending_mask"][:, :, 2:] = 0.0
        changed = clone_observation(original)
        changed["future_blocks"][:, 2:] = (
            1.0 - changed["future_blocks"][:, 2:]
        )
        changed["pending_blocks"][:, :, 2:] = (
            1.0 - changed["pending_blocks"][:, :, 2:]
        )

        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            extractor = extractor_class(space, features_dim=64).eval()
            with self.subTest(extractor=extractor_class.__name__), torch.no_grad():
                expected = extractor(original)
                actual = extractor(changed)
                torch.testing.assert_close(expected, actual)

    def test_all_zero_masks_make_all_padding_values_invariant(self):
        original = zero_observation()
        changed = clone_observation(original)
        changed["future_blocks"] = torch.linspace(
            0.0, 1.0, changed["future_blocks"].numel()
        ).reshape_as(changed["future_blocks"])
        changed["pending_blocks"] = torch.linspace(
            0.0, 1.0, changed["pending_blocks"].numel()
        ).reshape_as(changed["pending_blocks"])

        for extractor_class in EXTRACTOR_CLASSES:
            extractor = extractor_class(
                observation_space(grid_size=16), features_dim=64
            ).eval()
            with torch.no_grad():
                for network_name in (
                    "current",
                    "future",
                    "demand",
                    "pending",
                ):
                    for layer in getattr(
                        extractor.structured_encoder, network_name
                    ):
                        if isinstance(layer, torch.nn.Linear):
                            layer.bias.fill_(0.25)
            with self.subTest(
                extractor=extractor_class.__name__
            ), torch.no_grad():
                expected = extractor(original)
                actual = extractor(changed)
                torch.testing.assert_close(expected, actual)

    def test_fixed_grid_features_are_deterministic_adaptive_pooling(self):
        extractor = FixedGridExtractor(self.space, features_dim=64)
        grids = observation(batch_size=1)["grids"]

        actual = extractor._grid_features({"grids": grids})
        expected = F.adaptive_avg_pool2d(
            grids.flatten(0, 1), (8, 8)
        ).reshape(1, N_WORKSPACES, 256)

        torch.testing.assert_close(expected, actual)

    def test_structured_ignores_grids_while_grid_extractors_observe_them(self):
        space = observation_space(grid_size=16)
        original = observation(batch_size=1, grid_size=16)
        changed = clone_observation(original)
        changed["grids"].zero_()

        for extractor_class, should_change in (
            (StructuredExtractor, False),
            (FixedGridExtractor, True),
            (CandidateCnnExtractor, True),
        ):
            extractor = extractor_class(space, features_dim=64).eval()
            with self.subTest(extractor=extractor_class.__name__), torch.no_grad():
                differs = not torch.allclose(
                    extractor(original), extractor(changed)
                )
                self.assertEqual(should_change, differs)

    def test_candidate_cnn_uses_group_norm_and_four_channels(self):
        extractor = CandidateCnnExtractor(self.space)

        self.assertFalse(
            any(
                isinstance(module, torch.nn.BatchNorm2d)
                for module in extractor.modules()
            )
        )
        self.assertTrue(
            any(
                isinstance(module, torch.nn.GroupNorm)
                for module in extractor.modules()
            )
        )
        self.assertEqual(4, extractor.image_encoder[0].in_channels)

    def test_non_cnn_extractors_have_no_convolution(self):
        for extractor_class in (StructuredExtractor, FixedGridExtractor):
            extractor = extractor_class(self.space)
            self.assertFalse(
                any(
                    isinstance(module, torch.nn.Conv2d)
                    for module in extractor.modules()
                )
            )

    def test_candidate_cnn_exposes_workspace_image_features(self):
        extractor = CandidateCnnExtractor(self.space, features_dim=64)

        encoded = extractor.encode_grids(observation()["grids"])

        self.assertEqual((2, N_WORKSPACES, 128), tuple(encoded.shape))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_candidate_cnn_weights_receive_nonzero_update(self):
        extractor = CandidateCnnExtractor(self.space, features_dim=64)
        optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)
        before = extractor.image_encoder[0].weight.detach().clone()

        loss = extractor(observation()).square().mean()
        optimizer.zero_grad()
        loss.backward()

        self.assertGreater(
            extractor.image_encoder[0].weight.grad.norm().item(), 0.0
        )
        optimizer.step()
        self.assertFalse(
            torch.equal(before, extractor.image_encoder[0].weight)
        )

if __name__ == "__main__":
    unittest.main()
