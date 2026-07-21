import torch
import torch.nn as nn

from comparison.raw_direct_extractor import RawDirectExtractor
from test_feature_extractors import (
    clone_observation,
    observation,
    observation_space,
)


def expected_raw_concat(values):
    future_mask = values["future_mask"]
    pending_mask = values["pending_mask"]
    return torch.cat(
        (
            values["block"],
            values["ws_meta"].flatten(1),
            (values["future_blocks"] * future_mask.unsqueeze(-1)).flatten(1),
            future_mask.flatten(1),
            values["future_demand"].flatten(1),
            (values["pending_blocks"] * pending_mask.unsqueeze(-1)).flatten(1),
            pending_mask.flatten(1),
            values["pending_summary"].flatten(1),
        ),
        dim=1,
    )


def test_raw_direct_feature_dimension_and_order():
    values = observation(batch_size=1)
    extractor = RawDirectExtractor(observation_space())
    output = extractor(values)
    assert extractor.features_dim == 2818
    assert output.shape == (1, 2818)
    torch.testing.assert_close(output, expected_raw_concat(values))


def test_raw_direct_masks_invalid_slots():
    changed = observation(batch_size=1)
    changed["future_mask"][0, 3] = 0
    changed["future_blocks"][0, 3] = 1
    changed["pending_mask"][0, 2, 7] = 0
    changed["pending_blocks"][0, 2, 7] = 1
    expected = clone_observation(changed)
    expected["future_blocks"][0, 3] = 0
    expected["pending_blocks"][0, 2, 7] = 0
    extractor = RawDirectExtractor(observation_space())
    torch.testing.assert_close(extractor(changed), extractor(expected))


def test_raw_direct_ignores_grids_and_has_no_learned_layers():
    values = observation(batch_size=1)
    extractor = RawDirectExtractor(observation_space())
    changed = clone_observation(values)
    changed["grids"].fill_(1)
    torch.testing.assert_close(extractor(values), extractor(changed))
    assert not any(isinstance(m, (nn.Conv2d, nn.Linear)) for m in extractor.modules())
    assert sum(p.numel() for p in extractor.parameters()) == 0
