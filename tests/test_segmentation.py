import numpy as np

from segmentation import build_region_masks
from segmentation import create_mask_visualizations
from segmentation import summarize_label_map


def test_map_labels_groups_ade20k_ids_into_expected_regions():
    label_map = np.array(
        [
            [12, 2, 4, 0],
            [9, 17, 66, 1],
        ],
        dtype=np.int64,
    )

    masks = build_region_masks(label_map)

    assert masks["person"][0, 0]
    assert masks["sky"][0, 1]
    assert masks["vegetation"][0, 2]
    assert masks["vegetation"][1, 0]
    assert masks["background"][0, 3]


def test_create_mask_visualizations_returns_all_target_regions():
    label_map = np.zeros((2, 2), dtype=np.int64)
    result = create_mask_visualizations(label_map)
    assert set(result.keys()) == {"person", "sky", "vegetation", "background"}


def test_background_mask_is_complement_of_foreground_masks():
    label_map = np.array([[12, 2], [4, 0]], dtype=np.int64)
    masks = build_region_masks(label_map)
    foreground = masks["person"] | masks["sky"] | masks["vegetation"]
    assert np.array_equal(masks["background"], ~foreground)


def test_summarize_label_map_returns_sorted_top_labels():
    label_map = np.array(
        [
            [2, 2, 2, 12],
            [4, 4, 2, 12],
        ],
        dtype=np.int64,
    )
    summary = summarize_label_map(label_map, limit=2)
    assert summary[0]["label_id"] == 2
    assert summary[0]["pixel_count"] == 4
    assert summary[1]["label_id"] == 4


def test_sky_region_uses_checkpoint_label_id():
    label_map = np.array(
        [
            [2, 3],
            [0, 12],
        ],
        dtype=np.int64,
    )

    masks = build_region_masks(label_map)

    assert masks["sky"][0, 0]
    assert not masks["sky"][0, 1]
