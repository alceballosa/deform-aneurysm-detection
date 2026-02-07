"""
Comprehensive label consistency tests for DetectionCropper.

Verifies that annotation coordinates reported by DetectionCropper match the
actual spatial location of objects after the same spatial transform is applied.

Approach for each test case:
  1. Create a synthetic volume with solid boxes at known annotation locations.
  2. Pass the label volume as "vessel_edt" alongside the image so the cropper
     applies the exact same transform (matrix, spacing, rotation) to both.
  3. From the cropped label, extract the object center using scipy.ndimage.
  4. Compare that against the cropper's reported annotation center (ctr).

Test dimensions:
  - Crop sizes: 64, 128, 256
  - Number of lesions: 1, 2, 3+
  - Transforms: none, spacing only, rotation only, translation only, combined
  - Validation pipeline (no augmentation)
  - Various random seeds (200+ total cases)
"""

import atexit
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

PROJECT_ROOT = Path("/projects/vig/alberto/medical/exploration/deform")
REPORT_PATH = PROJECT_ROOT / "tests" / "cropper_label_report.tsv"
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.crop2 import DetectionCropper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_solid_box(shape, center_zyx, rad_zyx):
    """Create a binary volume with a solid box at given center/radii."""
    label = np.zeros(shape, dtype=np.float32)
    ctr = np.round(center_zyx).astype(int)
    half = np.round(np.array(rad_zyx) / 2).astype(int)

    z0 = max(0, ctr[0] - half[0])
    z1 = min(shape[0] - 1, ctr[0] + half[0])
    y0 = max(0, ctr[1] - half[1])
    y1 = min(shape[1] - 1, ctr[1] + half[1])
    x0 = max(0, ctr[2] - half[2])
    x1 = min(shape[2] - 1, ctr[2] + half[2])

    label[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return label


def make_sample(
    vol_shape,
    annot_locs,
    annot_rads,
    annot_cls=None,
    seed=42,
):
    """
    Build a sample dict with a random image and a label volume containing
    solid boxes at each annotation location.

    Args:
        vol_shape: (D, H, W) volume dimensions
        annot_locs: (N, 3) lesion centers in (z, y, x) order
        annot_rads: (N, 3) lesion radii in (d, h, w) order
        annot_cls: (N,) class labels; defaults to all-0 (aneurysm)
        seed: random seed for image generation

    Returns:
        dict ready for DetectionCropper.__call__
    """
    rng = np.random.RandomState(seed)
    image = rng.randn(*vol_shape).astype(np.float32)

    annot_locs = np.array(annot_locs, dtype=np.float32)
    annot_rads = np.array(annot_rads, dtype=np.float32)
    if annot_cls is None:
        annot_cls = np.zeros(len(annot_locs), dtype=np.int8)
    else:
        annot_cls = np.array(annot_cls, dtype=np.int8)

    # Build label volume with all annotations
    label = np.zeros(vol_shape, dtype=np.float32)
    for loc, rad in zip(annot_locs, annot_rads):
        label += draw_solid_box(vol_shape, loc, rad)
    label = np.clip(label, 0, 1)

    return {
        "scan_id": "test",
        "image": image,
        "image_spacing": (1.0, 1.0, 1.0),
        "all_loc": annot_locs,
        "all_rad": annot_rads,
        "all_cls": annot_cls,
        "vessel_edt": label,
    }


def extract_objects(label_crop, threshold=0.5):
    """
    Extract center-of-mass and bounding box size for each connected component.

    Returns list of dicts with keys 'com' (z,y,x) and 'size' (d,h,w),
    sorted by z coordinate.
    """
    binary = (label_crop > threshold).astype(np.int32)
    labeled, n_objects = ndimage.label(binary)
    objects = []
    if n_objects == 0:
        return objects
    slices_list = ndimage.find_objects(labeled)
    for obj_id in range(1, n_objects + 1):
        com = np.array(ndimage.center_of_mass(binary, labeled, obj_id))
        sl = slices_list[obj_id - 1]
        size = np.array([sl[ax].stop - sl[ax].start for ax in range(3)],
                        dtype=float)
        objects.append({"com": com, "size": size})
    objects.sort(key=lambda o: o["com"][0])
    return objects


def _is_near_boundary(ctr, rad, crop_shape, margin=2.0):
    """Check if an annotation's bounding box extends near/past the crop edge."""
    for axis in range(3):
        half = rad[axis] / 2.0 + margin
        if ctr[axis] - half < 0 or ctr[axis] + half > crop_shape[axis]:
            return True
    return False


# Global list to accumulate report rows across all tests
_REPORT_ROWS = []


def check_label_consistency(
    patches,
    atol_ctr=1.5,
    atol_rad=2.0,
    min_label_voxels=4,
    test_id="",
):
    """
    For every patch that has annotations, verify the cropper's reported
    center and radius match the actual object in the transformed label.

    Skips annotations whose bounding box extends past the crop boundary,
    since clipping shifts the center-of-mass and truncates the size.

    Args:
        patches: list of patch dicts from DetectionCropper
        atol_ctr: absolute tolerance in voxels for center comparison
        atol_rad: absolute tolerance in voxels for radius/size comparison
        min_label_voxels: minimum voxels in label to consider valid
        test_id: identifier for report logging

    Returns:
        (n_checked, n_passed, failures) where failures is a list of detail dicts
    """
    n_checked = 0
    n_passed = 0
    failures = []

    for i, patch in enumerate(patches):
        ctr = np.array(patch["ctr"])
        rad = np.array(patch["rad"])
        label_crop = patch["vessel_edt"][0]  # remove channel dim
        crop_shape = label_crop.shape

        if len(ctr) == 0:
            continue

        if label_crop.sum() < min_label_voxels:
            continue

        objects = extract_objects(label_crop)
        if len(objects) == 0:
            continue

        for j, reported_ctr in enumerate(ctr):
            reported_rad = rad[j] if j < len(rad) else np.array([5, 5, 5])
            if _is_near_boundary(reported_ctr, reported_rad, crop_shape):
                continue

            n_checked += 1

            # Find closest object by center distance
            dists = [np.linalg.norm(reported_ctr - o["com"]) for o in objects]
            best_idx = int(np.argmin(dists))
            best_obj = objects[best_idx]
            ctr_dist = dists[best_idx]

            # Compare size: reported_rad is half-extents, label size is full extent
            actual_size = best_obj["size"]
            rad_diff = np.abs(reported_rad - actual_size)
            max_rad_diff = float(np.max(rad_diff))

            ctr_ok = ctr_dist <= atol_ctr
            rad_ok = max_rad_diff <= atol_rad
            passed = ctr_ok and rad_ok

            # Log to report
            _REPORT_ROWS.append({
                "test_id": test_id,
                "patch": i,
                "annot": j,
                "crop_shape": crop_shape,
                "rep_ctr": np.round(reported_ctr, 2),
                "act_ctr": np.round(best_obj["com"], 2),
                "ctr_diff": np.round(reported_ctr - best_obj["com"], 2),
                "ctr_dist": round(ctr_dist, 3),
                "rep_rad": np.round(reported_rad, 2),
                "act_size": np.round(actual_size, 2),
                "rad_diff": np.round(rad_diff, 2),
                "max_rad_diff": round(max_rad_diff, 3),
                "ctr_ok": ctr_ok,
                "rad_ok": rad_ok,
                "passed": passed,
            })

            if passed:
                n_passed += 1
            else:
                failures.append({
                    "patch": i,
                    "annot": j,
                    "reported_ctr": reported_ctr,
                    "closest_com": best_obj["com"],
                    "dist": ctr_dist,
                    "reported_rad": reported_rad,
                    "actual_size": actual_size,
                    "rad_diff": rad_diff,
                    "max_rad_diff": max_rad_diff,
                    "ctr_ok": ctr_ok,
                    "rad_ok": rad_ok,
                })

    return n_checked, n_passed, failures


def write_report(path):
    """Write accumulated report rows to a TSV file."""
    if not _REPORT_ROWS:
        return
    with open(path, "w") as f:
        f.write(
            "test_id\tpatch\tannot\tcrop\t"
            "rep_ctr_z\trep_ctr_y\trep_ctr_x\t"
            "act_ctr_z\tact_ctr_y\tact_ctr_x\t"
            "ctr_dz\tctr_dy\tctr_dx\tctr_dist\t"
            "rep_rad_d\trep_rad_h\trep_rad_w\t"
            "act_size_d\tact_size_h\tact_size_w\t"
            "rad_dd\trad_dh\trad_dw\tmax_rad_diff\t"
            "ctr_ok\trad_ok\tpassed\n"
        )
        for r in _REPORT_ROWS:
            cs = r["crop_shape"]
            rc = r["rep_ctr"]
            ac = r["act_ctr"]
            cd = r["ctr_diff"]
            rr = r["rep_rad"]
            az = r["act_size"]
            rd = r["rad_diff"]
            f.write(
                f"{r['test_id']}\t{r['patch']}\t{r['annot']}\t"
                f"{cs[0]}x{cs[1]}x{cs[2]}\t"
                f"{rc[0]}\t{rc[1]}\t{rc[2]}\t"
                f"{ac[0]}\t{ac[1]}\t{ac[2]}\t"
                f"{cd[0]}\t{cd[1]}\t{cd[2]}\t{r['ctr_dist']}\t"
                f"{rr[0]}\t{rr[1]}\t{rr[2]}\t"
                f"{az[0]}\t{az[1]}\t{az[2]}\t"
                f"{rd[0]}\t{rd[1]}\t{rd[2]}\t{r['max_rad_diff']}\t"
                f"{r['ctr_ok']}\t{r['rad_ok']}\t{r['passed']}\n"
            )


# Write report at process exit
atexit.register(lambda: write_report(str(REPORT_PATH)))


def _format_failures(failures, n_checked):
    """Format failure details for assertion message."""
    lines = [f"{len(failures)}/{n_checked} annotations mismatched:"]
    for f in failures:
        parts = [
            f"  patch{f['patch']} annot{f['annot']}:",
            f"ctr reported={np.round(f['reported_ctr'], 1)}",
            f"actual={np.round(f['closest_com'], 1)}",
            f"dist={f['dist']:.2f}",
        ]
        if not f["ctr_ok"]:
            parts.append("[CTR FAIL]")
        parts.extend([
            f"| rad reported={np.round(f['reported_rad'], 1)}",
            f"actual_size={np.round(f['actual_size'], 1)}",
            f"max_diff={f['max_rad_diff']:.2f}",
        ])
        if not f["rad_ok"]:
            parts.append("[RAD FAIL]")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cropper configurations
# ---------------------------------------------------------------------------

def make_cropper(
    crop_size,
    rand_space=None,
    rand_rot=None,
    rand_trans=None,
    padded_reorient=False,
    sample_num=2,
    tp_ratio=1.0,
):
    """Helper to build a DetectionCropper with given params."""
    return DetectionCropper(
        crop_size=crop_size,
        rand_trans=rand_trans,
        rand_rot=rand_rot,
        rand_space=rand_space,
        spacing=[1.0, 1.0, 1.0],
        overlap=[16, 32, 32],
        tp_ratio=tp_ratio,
        sample_num=sample_num,
        blank_side=0,
        padded_reorient=padded_reorient,
        sample_cls=[0],
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Volume configs: (vol_shape, lesion definitions)
# Each lesion: (center_zyx, rad_zyx)
SINGLE_LESION_CONFIGS = [
    # Centered lesion
    {
        "vol_shape": (200, 200, 200),
        "locs": [[100, 100, 100]],
        "rads": [[13, 21, 19]],
    },
    # Off-center lesion
    {
        "vol_shape": (200, 200, 200),
        "locs": [[60, 140, 80]],
        "rads": [[10, 15, 12]],
    },
    # Small lesion
    {
        "vol_shape": (200, 200, 200),
        "locs": [[100, 100, 100]],
        "rads": [[5, 5, 5]],
    },
    # Large lesion
    {
        "vol_shape": (200, 200, 200),
        "locs": [[100, 100, 100]],
        "rads": [[25, 30, 28]],
    },
    # Near edge
    {
        "vol_shape": (200, 200, 200),
        "locs": [[30, 30, 30]],
        "rads": [[8, 8, 8]],
    },
]

MULTI_LESION_CONFIGS = [
    # Two lesions, close together
    {
        "vol_shape": (200, 200, 200),
        "locs": [[80, 100, 100], [120, 100, 100]],
        "rads": [[10, 10, 10], [8, 8, 8]],
    },
    # Two lesions, far apart
    {
        "vol_shape": (300, 300, 300),
        "locs": [[80, 80, 80], [220, 220, 220]],
        "rads": [[12, 15, 10], [10, 12, 14]],
    },
    # Three lesions
    {
        "vol_shape": (300, 300, 300),
        "locs": [[60, 60, 60], [150, 150, 150], [240, 240, 240]],
        "rads": [[10, 10, 10], [15, 15, 15], [8, 12, 10]],
    },
    # Two lesions, same z-slice
    {
        "vol_shape": (200, 300, 300),
        "locs": [[100, 80, 80], [100, 220, 220]],
        "rads": [[10, 12, 14], [8, 10, 10]],
    },
    # Three lesions, various sizes
    {
        "vol_shape": (250, 250, 250),
        "locs": [[50, 125, 125], [125, 50, 200], [200, 200, 50]],
        "rads": [[20, 15, 10], [6, 6, 6], [12, 18, 14]],
    },
]

CROP_SIZES = [
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
]

SPACING_RANGES = [
    [0.9, 1.2],
    [0.9, 1.8],
    [0.5, 1.5],
    [1.0, 2.0],
]

ROTATION_RANGES = [
    [5, 5, 0],
    [10, 10, 0],
    [15, 15, 15],
]

TRANSLATION_RANGES = [
    [3, 3, 3],
    [5, 5, 5],
    [10, 10, 10],
]


# ===== Test class: No augmentation (validation pipeline) =====

class TestNoAugmentation:
    """
    Validation pipeline: no spacing, rotation, or translation augmentation.
    The crop should match the annotation exactly.
    """

    @pytest.mark.parametrize("crop_size", CROP_SIZES, ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS + MULTI_LESION_CONFIGS,
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(3), ids=lambda s: f"seed{s}")
    def test_no_aug_label_consistency(self, crop_size, vol_cfg, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=None,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 7)
        patches = cropper(sample)

        tid = f"no_aug-crop{crop_size[0]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=1.5, atol_rad=3.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Spacing augmentation only =====

class TestSpacingOnly:
    """Spacing augmentation with no rotation or translation."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES, ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "spacing_range", SPACING_RANGES, ids=lambda s: f"sp{s[0]}_{s[1]}"
    )
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS[:3] + MULTI_LESION_CONFIGS[:3],
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_spacing_label_consistency(self, crop_size, spacing_range, vol_cfg, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 13)
        patches = cropper(sample)

        tid = f"spacing-crop{crop_size[0]}-sp{spacing_range[0]}_{spacing_range[1]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=2.0, atol_rad=5.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Rotation augmentation only =====

class TestRotationOnly:
    """Rotation augmentation with no spacing or translation."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES[:2], ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "rot_range", ROTATION_RANGES, ids=lambda r: f"rot{r[0]}_{r[1]}_{r[2]}"
    )
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS[:3] + MULTI_LESION_CONFIGS[:2],
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_rotation_label_consistency(self, crop_size, rot_range, vol_cfg, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=None,
            rand_rot=rot_range,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 17)
        patches = cropper(sample)

        # Higher tolerance for rotation (object shape changes slightly)
        # Rotation causes axis-aligned bbox expansion: size * (cos θ + sin θ)
        # For 15°, that's ~25% increase, so up to ~7 voxels for size-30 boxes
        tid = f"rotation-crop{crop_size[0]}-rot{rot_range[0]}_{rot_range[1]}_{rot_range[2]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=3.0, atol_rad=10.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Translation augmentation only =====

class TestTranslationOnly:
    """Translation augmentation with no spacing or rotation."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES[:2], ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "trans_range", TRANSLATION_RANGES, ids=lambda t: f"trans{t[0]}"
    )
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS[:3] + MULTI_LESION_CONFIGS[:2],
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_translation_label_consistency(
        self, crop_size, trans_range, vol_cfg, seed
    ):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=None,
            rand_rot=None,
            rand_trans=trans_range,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 23)
        patches = cropper(sample)

        tid = f"translation-crop{crop_size[0]}-trans{trans_range[0]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=1.5, atol_rad=3.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Combined augmentations =====

class TestCombinedAugmentations:
    """All augmentations active: spacing + rotation + translation."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES, ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "spacing_range", SPACING_RANGES[:2], ids=lambda s: f"sp{s[0]}_{s[1]}"
    )
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS[:3] + MULTI_LESION_CONFIGS[:3],
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_combined_label_consistency(
        self, crop_size, spacing_range, vol_cfg, seed
    ):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=[10, 10, 0],
            rand_trans=[5, 5, 5],
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 31)
        patches = cropper(sample)

        # Higher tolerance for combined augmentations
        tid = f"combined-crop{crop_size[0]}-sp{spacing_range[0]}_{spacing_range[1]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=3.5, atol_rad=12.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: padded_reorient=True (original behavior) =====

class TestPaddedReorientTrue:
    """Test with padded_reorient=True for comparison."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES[:2], ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "spacing_range",
        [[0.9, 1.2], [0.9, 1.8]],
        ids=lambda s: f"sp{s[0]}_{s[1]}",
    )
    @pytest.mark.parametrize(
        "vol_cfg",
        SINGLE_LESION_CONFIGS[:3] + MULTI_LESION_CONFIGS[:2],
        ids=lambda c: f"vol{c['vol_shape'][0]}_n{len(c['locs'])}",
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_padded_reorient_label_consistency(
        self, crop_size, spacing_range, vol_cfg, seed
    ):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=True,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=vol_cfg["vol_shape"],
            annot_locs=vol_cfg["locs"],
            annot_rads=vol_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 37)
        patches = cropper(sample)

        tid = f"padded_reorient-crop{crop_size[0]}-sp{spacing_range[0]}_{spacing_range[1]}-vol{vol_cfg['vol_shape'][0]}_n{len(vol_cfg['locs'])}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=2.0, atol_rad=5.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Output shape correctness =====

class TestOutputShape:
    """Verify padded_reorient=False always produces exactly crop_size output."""

    @pytest.mark.parametrize("crop_size", CROP_SIZES, ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "spacing_range", SPACING_RANGES, ids=lambda s: f"sp{s[0]}_{s[1]}"
    )
    @pytest.mark.parametrize("seed", range(5), ids=lambda s: f"seed{s}")
    def test_output_shape_exact(self, crop_size, spacing_range, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=3,
        )
        sample = make_sample(
            vol_shape=(200, 200, 200),
            annot_locs=[[100, 100, 100]],
            annot_rads=[[10, 10, 10]],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 41)
        patches = cropper(sample)

        for i, patch in enumerate(patches):
            img = patch["image"]
            # Shape is (1, D, H, W)
            assert img.shape == (1, crop_size[0], crop_size[1], crop_size[2]), (
                f"Patch {i}: expected (1,{crop_size[0]},{crop_size[1]},{crop_size[2]}) "
                f"got {img.shape}"
            )


# ===== Test class: Extreme spacing values =====

class TestExtremeSpacing:
    """Test with extreme spacing values to check robustness."""

    @pytest.mark.parametrize("crop_size", [[128, 128, 128]], ids=["crop128"])
    @pytest.mark.parametrize(
        "spacing_range",
        [[0.5, 0.6], [1.8, 2.0], [0.5, 2.0]],
        ids=["very_fine", "very_coarse", "wide_range"],
    )
    @pytest.mark.parametrize("seed", range(5), ids=lambda s: f"seed{s}")
    def test_extreme_spacing(self, crop_size, spacing_range, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=(300, 300, 300),
            annot_locs=[[150, 150, 150]],
            annot_rads=[[15, 20, 18]],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 43)
        patches = cropper(sample)

        tid = f"extreme_spacing-crop{crop_size[0]}-sp{spacing_range[0]}_{spacing_range[1]}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=2.5, atol_rad=6.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Asymmetric lesion radii =====

class TestAsymmetricRadii:
    """Test with highly asymmetric lesion dimensions."""

    ASYMMETRIC_CONFIGS = [
        {"locs": [[100, 100, 100]], "rads": [[5, 25, 10]]},
        {"locs": [[100, 100, 100]], "rads": [[25, 5, 5]]},
        {"locs": [[100, 100, 100]], "rads": [[8, 8, 30]]},
        {
            "locs": [[80, 100, 100], [130, 100, 100]],
            "rads": [[5, 20, 10], [20, 5, 15]],
        },
    ]

    @pytest.mark.parametrize("crop_size", CROP_SIZES[:2], ids=lambda c: f"crop{c[0]}")
    @pytest.mark.parametrize(
        "asym_cfg",
        ASYMMETRIC_CONFIGS,
        ids=[f"asym{i}" for i in range(len(ASYMMETRIC_CONFIGS))],
    )
    @pytest.mark.parametrize(
        "spacing_range",
        [None, [0.9, 1.5]],
        ids=["no_spacing", "spacing"],
    )
    @pytest.mark.parametrize("seed", range(2), ids=lambda s: f"seed{s}")
    def test_asymmetric_radii(self, crop_size, asym_cfg, spacing_range, seed):
        cropper = make_cropper(
            crop_size=crop_size,
            rand_space=spacing_range,
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=(200, 200, 200),
            annot_locs=asym_cfg["locs"],
            annot_rads=asym_cfg["rads"],
            seed=seed,
        )

        np.random.seed(seed * 1000 + 47)
        patches = cropper(sample)

        sp_str = f"sp{spacing_range[0]}_{spacing_range[1]}" if spacing_range else "no_sp"
        tid = f"asymmetric-crop{crop_size[0]}-{sp_str}-seed{seed}"
        n_checked, n_passed, failures = check_label_consistency(
            patches, atol_ctr=2.0, atol_rad=5.0, test_id=tid,
        )

        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)


# ===== Test class: Stress test with many random seeds =====

class TestStressRandomSeeds:
    """Run many random seeds to catch intermittent failures."""

    @pytest.mark.parametrize("seed", range(50), ids=lambda s: f"seed{s}")
    def test_stress_spacing_128(self, seed):
        """128 crop with random spacing, 50 seeds."""
        cropper = make_cropper(
            crop_size=[128, 128, 128],
            rand_space=[0.9, 1.8],
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=(200, 200, 200),
            annot_locs=[[100, 100, 100]],
            annot_rads=[[13, 21, 19]],
            seed=seed,
        )

        np.random.seed(seed)
        patches = cropper(sample)

        n_checked, _, failures = check_label_consistency(
            patches, atol_ctr=2.0, atol_rad=5.0, test_id=f"stress_spacing128-seed{seed}",
        )
        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)

    @pytest.mark.parametrize("seed", range(30), ids=lambda s: f"seed{s}")
    def test_stress_combined_128(self, seed):
        """128 crop with all augmentations, 30 seeds."""
        cropper = make_cropper(
            crop_size=[128, 128, 128],
            rand_space=[0.9, 1.5],
            rand_rot=[10, 10, 0],
            rand_trans=[5, 5, 5],
            padded_reorient=False,
            sample_num=2,
        )
        sample = make_sample(
            vol_shape=(250, 250, 250),
            annot_locs=[[125, 125, 125], [80, 80, 80]],
            annot_rads=[[13, 21, 19], [8, 10, 12]],
            seed=seed,
        )

        np.random.seed(seed + 500)
        patches = cropper(sample)

        n_checked, _, failures = check_label_consistency(
            patches, atol_ctr=3.5, atol_rad=12.0, test_id=f"stress_combined128-seed{seed}",
        )
        if n_checked == 0:
            pytest.skip("All annotations clipped at boundary")
        assert len(failures) == 0, _format_failures(failures, n_checked)

    @pytest.mark.parametrize("seed", range(20), ids=lambda s: f"seed{s}")
    def test_stress_multi_lesion_64(self, seed):
        """64 crop with 3 lesions and spacing, 20 seeds."""
        cropper = make_cropper(
            crop_size=[64, 64, 64],
            rand_space=[0.9, 1.5],
            rand_rot=None,
            rand_trans=None,
            padded_reorient=False,
            sample_num=3,
        )
        sample = make_sample(
            vol_shape=(200, 200, 200),
            annot_locs=[[60, 100, 100], [100, 60, 100], [100, 100, 60]],
            annot_rads=[[10, 10, 10], [8, 12, 8], [6, 6, 14]],
            seed=seed,
        )

        np.random.seed(seed + 800)
        patches = cropper(sample)

        n_checked, _, failures = check_label_consistency(
            patches, atol_ctr=2.0, atol_rad=5.0, test_id=f"stress_multi64-seed{seed}",
        )
        # It's OK if no annotations are in the crop for small 64-cubes
        if n_checked > 0:
            assert len(failures) == 0, _format_failures(failures, n_checked)
