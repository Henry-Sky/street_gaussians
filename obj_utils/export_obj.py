#!/usr/bin/env python3
"""
导出Waymo场景中动态对象

用法：
    python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 5
    python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# 确保项目根目录在sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 解析自定义参数
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--track_id', type=int, nargs='+', default=None)
parser.add_argument('--all', action='store_true')
parser.add_argument('--output_dir', type=str, default='./exports')
parser.add_argument('--metadata-only', action='store_true')
parser.add_argument('--no-pose', action='store_true')
parser.add_argument('--skip-ply', action='store_true')
args, remaining_argv = parser.parse_known_args()

sys.argv = [sys.argv[0]] + remaining_argv

from lib.config import cfg
from lib.datasets.dataset import sceneLoadTypeCallbacks
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.system_utils import searchForMaxIteration


def export_metadata(scene_info, track_ids, out_dir):
    """导出元数据JSON"""
    if 'obj_meta' not in scene_info.metadata:
        return []
    
    obj_meta = scene_info.metadata['obj_meta']
    if not isinstance(obj_meta, dict):
        return []
    
    d = os.path.join(out_dir, 'metadata')
    os.makedirs(d, exist_ok=True)
    files = []
    
    for tid in track_ids:
        if tid not in obj_meta:
            continue
        info = obj_meta[tid]
        # 序列化
        serializable = {}
        for k, v in info.items():
            if isinstance(v, (np.integer,)):
                serializable[k] = int(v)
            elif isinstance(v, (np.floating,)):
                serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v
        
        path = os.path.join(d, f'obj_{tid:03d}.json')
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        files.append(path)
        print(f"  ✓ {path}")
    
    return files


def export_trajectory(scene_info, track_ids, out_dir):
    """导出轨迹JSON"""
    if 'obj_tracklets' not in scene_info.metadata or 'obj_meta' not in scene_info.metadata:
        return []
    
    tracklets = scene_info.metadata['obj_tracklets']
    obj_meta = scene_info.metadata['obj_meta']
    
    if not isinstance(tracklets, np.ndarray) or tracklets.ndim != 3:
        return []
    
    d = os.path.join(out_dir, 'trajectories')
    os.makedirs(d, exist_ok=True)
    files = []
    
    nf, _, nfeat = tracklets.shape
    if nfeat < 8:
        return []
    
    for tid in track_ids:
        traj = []
        for f in range(nf):
            mask = tracklets[f, :, 0] == tid
            if not np.any(mask):
                continue
            col = np.where(mask)[0][0]
            pos = tracklets[f, col, 1:4].tolist()
            quat = tracklets[f, col, 4:8].tolist()
            
            # 计算偏航角
            qw, qx, qy, qz = quat
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))
            
            traj.append({
                "frame": int(f),
                "position": {"x": round(pos[0],4), "y": round(pos[1],4), "z": round(pos[2],4)},
                "orientation": {
                    "quaternion": [round(q,6) for q in quat],
                    "yaw_rad": round(yaw, 6),
                    "yaw_deg": round(np.degrees(yaw), 2)
                }
            })
        
        if not traj:
            continue
        
        info = obj_meta.get(tid, {})
        data = {
            "track_id": tid,
            "class": info.get('class', 'unknown'),
            "dimensions": {
                "length": float(info.get('length', 0)),
                "width": float(info.get('width', 0)),
                "height": float(info.get('height', 0))
            },
            "time_range": {
                "start_frame": traj[0]['frame'],
                "end_frame": traj[-1]['frame'],
                "duration_frames": len(traj)
            },
            "trajectory": traj
        }
        
        path = os.path.join(d, f'traj_{tid:03d}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        files.append(path)
        print(f"  ✓ {path} ({len(traj)}帧)")
    
    return files


def export_pth(tid, model_path, out_dir, include_pose=True):
    """导出PTH格式（高斯参数+位姿）"""
    cfg.model_path = model_path
    cfg.mode = 'evaluate'
    
    from lib.datasets.dataset import Dataset
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    # 优先从trained_model目录加载（完整模型）
    ckpt_dir = os.path.join(model_path, 'trained_model')
    if not os.path.exists(ckpt_dir):
        # 备选：尝试point_cloud目录
        ckpt_dir = os.path.join(model_path, 'point_cloud')
    
    loaded_iter = searchForMaxIteration(ckpt_dir)
    if not loaded_iter:
        print(f"⚠️ 未找到checkpoint")
        return None
    
    ckpt_path = os.path.join(ckpt_dir, f'iteration_{loaded_iter}.pth')
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 文件不存在: {ckpt_path}")
        return None
    
    print(f"📂 加载模型: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    gaussians.load_state_dict(state_dict)
    
    obj_name = f'obj_{tid:03d}'
    if not hasattr(gaussians, obj_name):
        print(f"⚠️ 对象 obj_{tid:03d} 不存在")
        return None
    
    obj = getattr(gaussians, obj_name)
    meta = gaussians.obj_info[tid]
    
    # 构建对象包
    data = {
        'gaussian_params': obj.state_dict(is_final=True),
        'obj_meta': {
            'track_id': int(meta['track_id']),
            'class': meta['class'],
            'class_label': meta['class_label'],
            'height': float(meta['height']),
            'width': float(meta['width']),
            'length': float(meta['length']),
            'deformable': meta['deformable'],
            'start_frame': int(meta['start_frame']),
            'end_frame': int(meta['end_frame']),
            'start_timestamp': float(meta['start_timestamp']),
            'end_timestamp': float(meta['end_timestamp']),
        },
        'fourier_config': {
            'fourier_dim': int(obj.fourier_dim),
            'fourier_scale': float(obj.fourier_scale),
        },
    }
    
    # 可选：位姿轨迹
    if include_pose and hasattr(gaussians, 'actor_pose'):

        try:
            track_idx = gaussians.actor_pose.obj_info[tid]['track_idx']
            pose = {'track_idx': track_idx.cpu().numpy().tolist()}
            if hasattr(gaussians.actor_pose, 'opt_trans'):
                pose['opt_trans'] = gaussians.actor_pose.opt_trans.detach().cpu().numpy().tolist()
            if hasattr(gaussians.actor_pose, 'opt_rots'):
                pose['opt_rots'] = gaussians.actor_pose.opt_rots.detach().cpu().numpy().tolist()
            data['pose_trajectory'] = pose
        except:
            pass
    
    d = os.path.join(out_dir, 'pth')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'obj_{tid:03d}.pth')
    torch.save(data, path)
    
    n_pts = obj.get_xyz.shape[0]
    print(f"  ✓ {path} ({n_pts}点)")
    return path


def export_ply(model_path, tid, out_dir):
    """导出PLY格式"""
    cfg.model_path = model_path
    cfg.mode = 'evaluate'
    
    from lib.datasets.dataset import Dataset
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    # 优先从trained_model目录加载
    ckpt_dir = os.path.join(model_path, 'trained_model')
    if not os.path.exists(ckpt_dir):
        # 备选：尝试point_cloud目录
        ckpt_dir = os.path.join(model_path, 'point_cloud')
    
    loaded_iter = searchForMaxIteration(ckpt_dir)
    if not loaded_iter:
        print(f"⚠️ 未找到checkpoint")
        return None, None
    
    ckpt_path = os.path.join(ckpt_dir, f'iteration_{loaded_iter}.pth')
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 文件不存在: {ckpt_path}")
        return None, None
    
    print(f"📂 加载模型: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    gaussians.load_state_dict(state_dict)
    
    obj_name = f'obj_{tid:03d}'
    if not hasattr(gaussians, obj_name):
        return None, None
    
    obj = getattr(gaussians, obj_name)
    
    d = os.path.join(out_dir, 'ply')
    os.makedirs(d, exist_ok=True)
    
    ply_path = os.path.join(d, f'obj_{tid:03d}.ply')
    obj.save_ply(ply_path)
    
    # 保存元数据
    meta = gaussians.obj_info[tid]
    meta_path = os.path.join(d, f'obj_{tid:03d}_meta.json')
    serializable = {}
    for k, v in meta.items():
        if isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    
    with open(meta_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"  ✓ {ply_path} ({obj.get_xyz.shape[0]}点)")
    return ply_path, meta_path


def main():
    if not cfg.source_path or not cfg.model_path:
        print("❌ 错误: 需要配置source_path和model_path")
        sys.exit(1)
    
    # 加载场景
    dataset_type = cfg.data.get('type', "Colmap")
    scene_info = sceneLoadTypeCallbacks[dataset_type](cfg.source_path, **cfg.data)
    
    # 确定对象列表
    if args.all and 'obj_meta' in scene_info.metadata:
        track_ids = list(scene_info.metadata['obj_meta'].keys())
    elif args.track_id:
        track_ids = args.track_id
    else:
        print("❌ 错误: 请指定--track_id或--all")
        sys.exit(1)
    
    waymo_id = os.path.basename(cfg.source_path)
    out_dir = os.path.join(args.output_dir, waymo_id)
    
    print("=" * 60)
    print("📦 导出工具")
    print(f"数据: {waymo_id}, 对象: {len(track_ids)}个")
    print(f"模式: {'仅元数据' if args.metadata_only else '完整'}")
    print("=" * 60)
    
    try:
        # 1. 元数据
        print("\n📝 [1/4] 元数据...")
        export_metadata(scene_info, track_ids, out_dir)
        
        # 2. 轨迹
        print("\n🛤️  [2/4] 轨迹...")
        export_trajectory(scene_info, track_ids, out_dir)
        
        if args.metadata_only:
            print("\n✅ 元数据导出完成")
            return
        
        # 3. PTH
        print("\n🗜️  [3/4] PTH...")
        for tid in track_ids:
            export_pth(tid, cfg.model_path, out_dir, not args.no_pose)
        
        # 4. PLY
        if not args.skip_ply:
            print("\n☁️  [4/4] PLY...")
            for tid in track_ids:
                export_ply(tid, cfg.model_path, out_dir)

        print("\n" + "=" * 60)
        print("✅ 导出完成！")
        print(f"目录: {out_dir}/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
