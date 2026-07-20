import unittest

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)


N_WORKSPACES = 10


def observation_space(grid_size: int = 64) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (N_WORKSPACES, 4), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 4), np.float32),
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
        "ws_meta": torch.rand(batch_size, N_WORKSPACES, 4),
        "future_blocks": torch.rand(batch_size, 16, 6),
        "future_mask": torch.ones(batch_size, 16),
        "future_demand": torch.rand(batch_size, 3, 4),
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
            "demand": [(12, 64), (64, 32)],
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
            (StructuredExtractor, 200),
            (FixedGridExtractor, 456),
            (CandidateCnnExtractor, 328),
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

    def test_every_extractor_observes_each_corrected_structured_key(self):
        space = observation_space(grid_size=16)
        original = observation(batch_size=1, grid_size=16)

        def change_block(changed):
            changed["block"][:, 0] += 1.0

        def change_future(changed):
            changed["future_blocks"][:, 0, 0] += 1.0

        def change_future_mask(changed):
            changed["future_mask"][:, 0] = 0.0

        def change_demand(changed):
            changed["future_demand"][:, 0, 0] += 1.0

        def change_pending(changed):
            changed["pending_blocks"][:, 0, 0, 0] += 1.0

        def change_pending_mask(changed):
            changed["pending_mask"][:, 0, 0] = 0.0

        def change_pending_summary(changed):
            changed["pending_summary"][:, 0, 0] += 1.0

        def change_ws_meta(changed):
            changed["ws_meta"][:, 0, 0] += 1.0

        mutations = {
            "block": change_block,
            "future_blocks": change_future,
            "future_mask": change_future_mask,
            "future_demand": change_demand,
            "pending_blocks": change_pending,
            "pending_mask": change_pending_mask,
            "pending_summary": change_pending_summary,
            "ws_meta": change_ws_meta,
        }

        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            extractor = extractor_class(space, features_dim=64).eval()
            with torch.no_grad():
                baseline = extractor(original)
                for key, mutate in mutations.items():
                    with self.subTest(
                        extractor=extractor_class.__name__, key=key
                    ):
                        changed = clone_observation(original)
                        mutate(changed)
                        self.assertFalse(
                            torch.allclose(baseline, extractor(changed))
                        )

    def test_future_and_pending_sequence_order_changes_features(self):
        space = observation_space(grid_size=16)
        original = observation(batch_size=1, grid_size=16)

        future_swapped = clone_observation(original)
        future_swapped["future_blocks"][:, [0, 1]] = future_swapped[
            "future_blocks"
        ][:, [1, 0]]
        pending_swapped = clone_observation(original)
        pending_swapped["pending_blocks"][:, 0, [0, 1]] = pending_swapped[
            "pending_blocks"
        ][:, 0, [1, 0]]

        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            extractor = extractor_class(space, features_dim=64).eval()
            with self.subTest(extractor=extractor_class.__name__), torch.no_grad():
                baseline = extractor(original)
                self.assertFalse(
                    torch.allclose(baseline, extractor(future_swapped))
                )
                self.assertFalse(
                    torch.allclose(baseline, extractor(pending_swapped))
                )

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

    def test_workspace_order_is_preserved_in_final_projection(self):
        extractor = StructuredExtractor(self.space, features_dim=64).eval()
        original = observation(batch_size=1)
        swapped = clone_observation(original)
        for key in (
            "ws_meta",
            "pending_blocks",
            "pending_mask",
            "pending_summary",
        ):
            swapped[key][:, [0, 1]] = swapped[key][:, [1, 0]]

        with torch.no_grad():
            self.assertFalse(
                torch.allclose(extractor(original), extractor(swapped))
            )

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

    def test_schema_validation_rejects_missing_and_legacy_shapes(self):
        missing = observation_space()
        del missing.spaces["pending_summary"]
        with self.assertRaisesRegex(ValueError, "nine keys|pending_summary"):
            StructuredExtractor(missing)

        legacy = observation_space()
        legacy.spaces["block"] = gym.spaces.Box(
            0, 1, (10,), np.float32
        )
        with self.assertRaisesRegex(ValueError, "block.*8"):
            StructuredExtractor(legacy)

        inconsistent = observation_space()
        inconsistent.spaces["pending_mask"] = gym.spaces.Box(
            0, 1, (9, 32), np.float32
        )
        with self.assertRaisesRegex(ValueError, "pending_mask.*10"):
            StructuredExtractor(inconsistent)


if __name__ == "__main__":
    unittest.main()
