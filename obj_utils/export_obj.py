#!/usr/bin/env python3
"""
导出Waymo场景中动态对象

新目录结构：
    exports/031/obj_011/
        ├── metadata.json      # 元数据
        ├── trajectory.json    # 轨迹数据
        ├── model.pth          # 模型参数（高斯点云+位姿）
        └── point_cloud.ply    # 点云文件

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


def export_object_data(scene_info, track_ids, model_path, out_dir, include_pose=True, skip_ply=False):
    """
    按对象ID组织导出数据
    每个对象一个目录：out_dir/obj_{tid:03d}/
    """
    if 'obj_meta' not in scene_info.metadata:
        print("⚠️ 场景中没有对象元数据")
        return []
    
    obj_meta = scene_info.metadata['obj_meta']
    if not isinstance(obj_meta, dict):
        print("⚠️ 对象元数据格式错误")
        return []
    
    # 加载模型（所有对象共享一次加载）
    cfg.model_path = model_path
    cfg.mode = 'evaluate'
    
    from lib.datasets.dataset import Dataset
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    # 优先从trained_model目录加载（完整模型）
    ckpt_dir = os.path.join(model_path, 'trained_model')
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(model_path, 'point_cloud')
    
    loaded_iter = searchForMaxIteration(ckpt_dir)
    if not loaded_iter:
        print(f"⚠️ 未找到checkpoint")
        return []
    
    ckpt_path = os.path.join(ckpt_dir, f'iteration_{loaded_iter}.pth')
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 文件不存在: {ckpt_path}")
        return []
    
    print(f"📂 加载模型: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    gaussians.load_state_dict(state_dict)
    
    exported_objects = []
    
    for tid in track_ids:
        if tid not in obj_meta:
            print(f"⚠️ 对象 {tid} 不存在于元数据中，跳过")
            continue
        
        # 创建对象目录
        obj_dir = os.path.join(out_dir, f'obj_{tid:03d}')
        os.makedirs(obj_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📦 导出对象: obj_{tid:03d}")
        print(f"{'='*60}")
        
        info = obj_meta[tid]
        obj_name = f'obj_{tid:03d}'
        
        if not hasattr(gaussians, obj_name):
            print(f"⚠️ 对象 {obj_name} 在模型中不存在，跳过")
            continue
        
        obj = getattr(gaussians, obj_name)
        files_exported = []
        
        # 1. 导出元数据 JSON
        try:
            meta_path = os.path.join(obj_dir, 'metadata.json')
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
            
            with open(meta_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            files_exported.append(meta_path)
            print(f"  ✓ metadata.json")
        except Exception as e:
            print(f"  ✗ metadata.json 失败: {e}")
        
        # 2. 导出轨迹 JSON
        try:
            if 'obj_tracklets' in scene_info.metadata:
                tracklets = scene_info.metadata['obj_tracklets']
                if isinstance(tracklets, np.ndarray) and tracklets.ndim == 3:
                    traj = []
                    nf, _, nfeat = tracklets.shape
                    if nfeat >= 8:
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
                        
                        if traj:
                            traj_data = {
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
                            
                            traj_path = os.path.join(obj_dir, 'trajectory.json')
                            with open(traj_path, 'w') as f:
                                json.dump(traj_data, f, indent=2)
                            files_exported.append(traj_path)
                            print(f"  ✓ trajectory.json ({len(traj)}帧)")
        except Exception as e:
            print(f"  ✗ trajectory.json 失败: {e}")
        
        # 3. 导出 PTH 模型参数
        try:
            meta = gaussians.obj_info[tid]
            
            # 构建对象包
            pth_data = {
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
                    pth_data['pose_trajectory'] = pose
                except Exception as e:
                    print(f"  ⚠️ 位姿信息导出警告: {e}")
            
            pth_path = os.path.join(obj_dir, 'model.pth')
            torch.save(pth_data, pth_path)
            files_exported.append(pth_path)
            n_pts = obj.get_xyz.shape[0]
            print(f"  ✓ model.pth ({n_pts}个高斯点)")
        except Exception as e:
            print(f"  ✗ model.pth 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. 导出 PLY 点云
        if not skip_ply:
            try:
                ply_path = os.path.join(obj_dir, 'point_cloud.ply')
                obj.save_ply(ply_path)
                files_exported.append(ply_path)
                print(f"  ✓ point_cloud.ply")
            except Exception as e:
                print(f"  ✗ point_cloud.ply 失败: {e}")
        
        exported_objects.append({
            'track_id': tid,
            'directory': obj_dir,
            'files': files_exported
        })
    
    return exported_objects


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
    print("📦 对象导出工具 v2.0")
    print(f"数据: {waymo_id}")
    print(f"对象数量: {len(track_ids)}个")
    print(f"输出目录: {out_dir}/")
    print(f"模式: {'仅元数据' if args.metadata_only else '完整'}")
    print("=" * 60)
    
    try:
        if args.metadata_only:
            # 仅导出元数据（保持向后兼容）
            obj_meta = scene_info.metadata.get('obj_meta', {})
            for tid in track_ids:
                if tid not in obj_meta:
                    continue
                obj_dir = os.path.join(out_dir, f'obj_{tid:03d}')
                os.makedirs(obj_dir, exist_ok=True)
                
                info = obj_meta[tid]
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
                
                meta_path = os.path.join(obj_dir, 'metadata.json')
                with open(meta_path, 'w') as f:
                    json.dump(serializable, f, indent=2)
                print(f"  ✓ obj_{tid:03d}/metadata.json")
            
            print("\n✅ 元数据导出完成")
            return
        
        # 完整导出
        exported = export_object_data(
            scene_info, 
            track_ids, 
            cfg.model_path, 
            out_dir,
            include_pose=not args.no_pose,
            skip_ply=args.skip_ply
        )
        
        # 统计结果
        total_files = sum(len(obj['files']) for obj in exported)
        
        print("\n" + "=" * 60)
        print("✅ 导出完成！")
        print(f"成功导出对象: {len(exported)}/{len(track_ids)}")
        print(f"总文件数: {total_files}")
        print(f"目录结构: {out_dir}/")
        print("\n目录示例:")
        if exported:
            sample_obj = exported[0]
            print(f"  {sample_obj['directory']}/")
            for f in sample_obj['files']:
                print(f"    ├── {os.path.basename(f)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
