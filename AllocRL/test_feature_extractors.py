import io
import inspect
import unittest
import warnings

import gymnasium as gym
import numpy as np
import torch

from alloc_env.alloc_env import FUTURE_BLOCK_FEATURE_DIM
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)


def observation_space(
    n_workspaces: int = 3,
    grid_size: int = 32,
    n_future_blocks: int = 4,
) -> gym.spaces.Dict:
    spaces = {
        "block": gym.spaces.Box(
            0.0, 1.0, shape=(10,), dtype=np.float32
        ),
        "grids": gym.spaces.Box(
            0.0,
            1.0,
            shape=(n_workspaces, 4, grid_size, grid_size),
            dtype=np.float32,
        ),
        "ws_meta": gym.spaces.Box(
            0.0, 1.0, shape=(n_workspaces, 3), dtype=np.float32
        ),
    }
    if n_future_blocks > 0:
        spaces["future_blocks"] = gym.spaces.Box(
            0.0,
            1.0,
            shape=(n_future_blocks, FUTURE_BLOCK_FEATURE_DIM),
            dtype=np.float32,
        )
        spaces["future_mask"] = gym.spaces.Box(
            0.0,
            1.0,
            shape=(n_future_blocks,),
            dtype=np.float32,
        )
    return gym.spaces.Dict(spaces)


def observation(
    batch_size: int = 2,
    n_workspaces: int = 3,
    grid_size: int = 32,
    n_future_blocks: int = 4,
) -> dict[str, torch.Tensor]:
    result = {
        "block": torch.rand(batch_size, 10),
        "grids": torch.rand(
            batch_size, n_workspaces, 4, grid_size, grid_size
        ),
        "ws_meta": torch.rand(batch_size, n_workspaces, 3),
    }
    if n_future_blocks > 0:
        result["future_blocks"] = torch.rand(
            batch_size, n_future_blocks, FUTURE_BLOCK_FEATURE_DIM
        )
        result["future_mask"] = torch.ones(
            batch_size, n_future_blocks
        )
    return result


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.space = observation_space()

    def test_all_extractors_return_finite_features(self):
        obs = observation()
        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            with self.subTest(extractor=extractor_class.__name__):
                extractor = extractor_class(self.space, features_dim=64)
                features = extractor(obs)
                self.assertEqual((2, 64), tuple(features.shape))
                self.assertTrue(torch.isfinite(features).all())

    def test_extractors_support_observations_without_future_blocks(self):
        space = observation_space(n_future_blocks=0)
        obs = observation(n_future_blocks=0)
        for extractor_class in (
            StructuredExtractor,
            FixedGridExtractor,
            CandidateCnnExtractor,
        ):
            with self.subTest(extractor=extractor_class.__name__):
                features = extractor_class(space, features_dim=32)(obs)
                self.assertEqual((2, 32), tuple(features.shape))

    def test_future_order_changes_features(self):
        extractor = StructuredExtractor(self.space, features_dim=64)
        obs = observation()
        swapped = {key: value.clone() for key, value in obs.items()}
        swapped["future_blocks"][:, [0, 1]] = swapped[
            "future_blocks"
        ][:, [1, 0]]

        with torch.no_grad():
            first = extractor(obs)
            second = extractor(swapped)

        self.assertFalse(torch.allclose(first, second))

    def test_padding_values_do_not_change_features(self):
        extractor = CandidateCnnExtractor(self.space, features_dim=64).eval()
        obs = observation()
        obs["future_mask"][:, 2:] = 0.0
        changed = {key: value.clone() for key, value in obs.items()}
        changed["future_blocks"][:, 2:] = 999.0

        with torch.no_grad():
            expected = extractor(obs)
            actual = extractor(changed)

        torch.testing.assert_close(expected, actual)

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

        self.assertEqual((2, 3, 128), tuple(encoded.shape))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_candidate_cnn_exports_to_onnx_with_dynamic_batch(self):
        class GridEncoder(torch.nn.Module):
            def __init__(self, space):
                super().__init__()
                self.extractor = CandidateCnnExtractor(
                    space, features_dim=64
                ).eval()

            def forward(self, grids):
                return self.extractor.encode_grids(grids)

        for grid_size in (16, 32, 64):
            with self.subTest(grid_size=grid_size):
                space = observation_space(grid_size=grid_size)
                output = io.BytesIO()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", torch.jit.TracerWarning)
                    export_kwargs = {}
                    if "dynamo" in inspect.signature(
                        torch.onnx.export
                    ).parameters:
                        export_kwargs["dynamo"] = False
                    torch.onnx.export(
                        GridEncoder(space),
                        observation(
                            batch_size=1, grid_size=grid_size
                        )["grids"],
                        output,
                        input_names=["grids"],
                        output_names=["workspace_features"],
                        dynamic_axes={
                            "grids": {0: "batch"},
                            "workspace_features": {0: "batch"},
                        },
                        opset_version=17,
                        **export_kwargs,
                    )

                self.assertGreater(len(output.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
