"""Swap two dynamic-object appearances in Waymo scene 002 and render the result.

Usage:
    python main.py --config configs/example/waymo_train_002.yaml

The source model is the model selected by ``--config``.  The swapped checkpoint
and its trajectory renders are written below
``output/waymo_full_exp/waymo_swap_002``.  Object poses are deliberately left
unchanged, so each swapped appearance follows the original object's trajectory.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

# Importing config parses ``gpus`` and sets CUDA_VISIBLE_DEVICES.  It must be
# done before importing torch, otherwise CUDA can reject the late environment
# change during its first initialization.
from lib.config import cfg

import torch
from tqdm import tqdm

from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import safe_state
from lib.visualizers.base_visualizer import BaseVisualizer


SCENE_ID = "002"
SWAP_TRACK_IDS = (0, 4)
OUTPUT_ROOT = Path("output/waymo_full_exp/waymo_swap_002")


def _checkpoint_iteration(checkpoint_path: Path) -> int:
    """Read the iteration from a standard ``iteration_<n>.pth`` filename."""
    return int(checkpoint_path.stem.replace("iteration_", "", 1))


def _prepare_swapped_checkpoint(source_root: Path, target_root: Path) -> int:
    """Copy a checkpoint and exchange the complete Gaussian point sets of two actors."""
    source_checkpoint_dir = source_root / "trained_model"
    source_checkpoints = sorted(source_checkpoint_dir.glob("iteration_*.pth"))
    if not source_checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {source_checkpoint_dir}")

    source_checkpoint = max(source_checkpoints, key=_checkpoint_iteration)
    iteration = _checkpoint_iteration(source_checkpoint)
    target_checkpoint = target_root / "trained_model" / source_checkpoint.name
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # GaussianModel.load_state_dict assigns tensors directly (it does not call
    # .cuda()). Preserve the source checkpoint device so the rasterizer receives
    # CUDA tensors instead of silently receiving CPU Gaussian buffers.
    checkpoint_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(source_checkpoint, map_location=checkpoint_device, weights_only=False)
    object_a, object_b = (f"obj_{track_id:03d}" for track_id in SWAP_TRACK_IDS)
    missing = [name for name in (object_a, object_b) if name not in state]
    if missing:
        available = sorted(name for name in state if name.startswith("obj_"))
        raise KeyError(f"Cannot swap {missing}; checkpoint objects: {available}")

    # Exchange complete Gaussian point sets.  actor_pose is deliberately kept
    # untouched: the swapped point set follows the destination track's motion.
    state[object_a], state[object_b] = (
        copy.deepcopy(state[object_b]),
        copy.deepcopy(state[object_a]),
    )
    torch.save(state, target_checkpoint)

    # Scene only uses point_cloud to discover the load iteration in eval mode,
    # but retaining the PLY keeps the swapped output self-contained.
    source_ply = source_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    target_ply = target_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    target_ply.parent.mkdir(parents=True, exist_ok=True)
    if source_ply.exists():
        shutil.copy2(source_ply, target_ply)
    else:
        raise FileNotFoundError(f"Expected point cloud is missing: {source_ply}")

    manifest = {
        "source_model": str(source_root),
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(target_checkpoint),
        "iteration": iteration,
        "swapped_track_ids": list(SWAP_TRACK_IDS),
        "swap_policy": "complete Gaussian point sets exchanged; actor_pose unchanged.",
    }
    with (target_root / "swap_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return iteration


def _reuse_source_scene_assets(source_root: Path, target_root: Path) -> None:
    """Link immutable preprocessing assets so rendering does not rerun COLMAP."""
    for asset_name in ("input_ply", "colmap"):
        source_asset = source_root / asset_name
        target_asset = target_root / asset_name
        if not source_asset.exists():
            raise FileNotFoundError(f"Source scene asset is missing: {source_asset}")

        if target_asset.is_symlink():
            if target_asset.resolve() == source_asset.resolve():
                continue
            raise FileExistsError(f"Refusing to replace unrelated link: {target_asset}")
        if target_asset.exists():
            # A complete directory may be a user's earlier output.  Only remove
            # the incomplete COLMAP directory created by an interrupted run.
            if asset_name == "colmap" and not (target_asset / "triangulated/sparse/model").is_dir():
                shutil.rmtree(target_asset)
            else:
                continue
        os.symlink(source_asset, target_asset, target_is_directory=True)


def _render_trajectory() -> Path:
    """Render the swapped composition along the original camera trajectory."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Gaussian rasterization, but no usable CUDA device is available. "
            "The swapped checkpoint has still been written and can be rendered on a GPU host."
        )
    cfg.render.save_image = False
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        output_dir = Path(cfg.model_path) / "trajectory" / f"ours_{scene.loaded_iter}"
        visualizer = BaseVisualizer(str(output_dir))
        # BaseVisualizer expects this collection when it groups multi-camera
        # video frames, but only StreetGaussianVisualizer normally populates it.
        visualizer.cams = []

        cameras = sorted(
            scene.getTrainCameras() + scene.getTestCameras(), key=lambda camera: camera.id
        )
        for camera in tqdm(cameras, desc="Rendering swapped trajectory"):
            visualizer.cams.append(camera.meta['cam'])
            result = renderer.render(camera, gaussians)
            # The renderer's no-visible-Gaussian fast path omits depth, whereas
            # BaseVisualizer always writes a depth video.
            if "depth" not in result:
                result["depth"] = torch.zeros(
                    1, int(camera.image_height), int(camera.image_width), device="cuda"
                )
            visualizer.visualize(result, camera)
        # This is an appearance-swap render: retain the required color video.
        # Empty frames can make the optional depth/diff color maps undefined.
        visualizer.depths.clear()
        visualizer.diffs.clear()
        visualizer.summarize()
    return output_dir


def main() -> None:
    safe_state(cfg.eval.quiet)
    source_root = Path(cfg.model_path).resolve()
    target_root = (Path(cfg.workspace) / OUTPUT_ROOT).resolve()

    if source_root == target_root:
        raise ValueError("The configured source model must not be the swap output directory.")
    if SCENE_ID not in Path(cfg.source_path).parts:
        raise ValueError(f"This test is fixed to scene {SCENE_ID}; got source_path={cfg.source_path}")

    iteration = _prepare_swapped_checkpoint(source_root, target_root)
    _reuse_source_scene_assets(source_root, target_root)

    # Scene derives these paths once during configuration parsing, so update all
    # of them before Dataset/Scene are created.
    cfg.model_path = str(target_root)
    cfg.trained_model_dir = str(target_root / "trained_model")
    cfg.point_cloud_dir = str(target_root / "point_cloud")
    cfg.loaded_iter = iteration
    cfg.mode = "trajectory"

    render_dir = _render_trajectory()
    print(f"Swapped obj_{SWAP_TRACK_IDS[0]:03d} <-> obj_{SWAP_TRACK_IDS[1]:03d}")
    print(f"Checkpoint: {target_root / 'trained_model' / f'iteration_{iteration}.pth'}")
    print(f"Renders: {render_dir}")


if __name__ == "__main__":
    main()
