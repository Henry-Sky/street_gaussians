#!/usr/bin/env python3
"""
查询Waymo场景中动态对象信息

用法：
    python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml
    python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml --track_id 5 --verbose
"""

import os
import sys
import argparse
import numpy as np

# 确保项目根目录在sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 解析自定义参数
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--track_id', type=int, nargs='+', default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--mode', type=str, choices=['metadata', 'trajectory', 'both'], default='both')
args, remaining_argv = parser.parse_known_args()

# 重置sys.argv供cfg使用
sys.argv = [sys.argv[0]] + remaining_argv

from lib.config import cfg
from lib.datasets.dataset import sceneLoadTypeCallbacks


def print_summary(scene_info, track_ids=None):
    """打印对象摘要"""
    metadata = scene_info.metadata
    if 'obj_meta' not in metadata:
        print("⚠️ 没有对象元数据")
        return
    
    obj_meta = metadata['obj_meta']
    if not isinstance(obj_meta, dict):
        print(f"⚠️ obj_meta格式不支持")
        return
    
    # 确定显示的对象
    if track_ids:
        display_ids = [tid for tid in track_ids if tid in obj_meta]
    else:
        display_ids = list(obj_meta.keys())
    
    if not display_ids:
        print("⚠️ 未找到指定对象")
        return
    
    # 类别统计
    class_stats = {}
    for tid in display_ids:
        cls = obj_meta[tid].get('class', 'unknown')
        class_stats.setdefault(cls, []).append(tid)
    
    print("=" * 60)
    print(f"📊 动态对象摘要 ({len(display_ids)}个对象)")
    print("=" * 60)
    print("\n📊 类别统计:")
    for cls, ids in sorted(class_stats.items()):
        print(f"  - {cls.upper()}: {len(ids)}个 (IDs: {ids[:5]}{'...' if len(ids)>5 else ''})")
    
    print(f"\n{'ID':<4} {'类别':<10} {'尺寸(m)':<18} {'生命周期':<15}")
    print("-" * 60)
    
    for tid in sorted(display_ids):
        info = obj_meta[tid]
        label = info.get('class_label', info.get('class', 'N/A'))
        l, w, h = info.get('length', 0), info.get('width', 0), info.get('height', 0)
        sf, ef = info.get('start_frame', 'N/A'), info.get('end_frame', 'N/A')
        dur = f"{sf}-{ef}" if sf != 'N/A' else "N/A"
        print(f"{tid:<4} {label:<10} {l:.2f}×{w:.2f}×{h:.2f}{'':<6} {dur:<15}")
    
    print("=" * 60)


def print_detail(scene_info, track_ids=None):
    """打印详细信息"""
    metadata = scene_info.metadata
    if 'obj_meta' not in metadata:
        return
    
    obj_meta = metadata['obj_meta']
    if not isinstance(obj_meta, dict):
        return
    
    if track_ids:
        display_ids = [tid for tid in track_ids if tid in obj_meta]
    else:
        display_ids = list(obj_meta.keys())
    
    print("=" * 60)
    print(f"🔍 详细信息 ({len(display_ids)}个对象)")
    print("=" * 60)
    
    for tid in sorted(display_ids):
        info = obj_meta[tid]
        print(f"\n━━━ 对象 {tid} ━━━")
        print(f"  类别: {info.get('class', 'N/A')}")
        print(f"  标签: {info.get('class_label', 'N/A')}")
        print(f"  尺寸: {info.get('length', 0):.3f}×{info.get('width', 0):.3f}×{info.get('height', 0):.3f}m")
        print(f"  时间: 帧{info.get('start_frame', '?')}-{info.get('end_frame', '?')}")
        if 'start_timestamp' in info:
            print(f"  时长: {float(info['end_timestamp'])-float(info['start_timestamp']):.2f}s")


def print_trajectory(scene_info, track_ids=None):
    """打印轨迹信息"""
    metadata = scene_info.metadata
    if 'obj_tracklets' not in metadata:
        print("⚠️ 没有轨迹数据")
        return
    
    tracklets = metadata['obj_tracklets']
    if not isinstance(tracklets, np.ndarray) or tracklets.ndim != 3:
        print("⚠️ 轨迹格式错误")
        return
    
    # 获取有效ID
    valid_mask = tracklets[:, :, 0] >= 0
    unique_ids = np.unique(tracklets[:,:,0][valid_mask])
    if track_ids:
        unique_ids = np.array([tid for tid in track_ids if tid in unique_ids])
    
    print("=" * 60)
    print(f"🛤️  轨迹信息 ({len(unique_ids)}个对象, {tracklets.shape[0]}帧)")
    print("=" * 60)
    
    for tid in sorted(unique_ids):
        mask = tracklets[:, :, 0] == tid
        frames = np.where(np.any(mask, axis=1))[0]
        if len(frames) == 0:
            continue
        
        # 计算移动距离
        positions = []
        for f in frames:
            col = np.where(mask[f])[0][0]
            positions.append(tracklets[f, col, 1:4])
        
        dist = np.linalg.norm(positions[-1] - positions[0]) if len(positions) > 1 else 0
        speed = dist / (len(frames)/10.0) if len(frames) > 1 else 0
        
        print(f"ID {tid}: {len(frames)}帧, 移动{dist:.2f}m, 速度{speed:.2f}m/s")


def main():
    if not cfg.source_path:
        print("❌ 错误: 未配置source_path")
        sys.exit(1)
    
    print(f"📂 {cfg.source_path}")
    
    try:
        # 加载元数据（不加载图像，避免OOM）
        dataset_type = cfg.data.get('type', "Colmap")
        scene_info = sceneLoadTypeCallbacks[dataset_type](cfg.source_path, **cfg.data)
        
        if args.mode in ['metadata', 'both']:
            if args.verbose:
                print_detail(scene_info, args.track_id)
            else:
                print_summary(scene_info, args.track_id)
        
        if args.mode in ['trajectory', 'both']:
            print_trajectory(scene_info, args.track_id)
        
        print("\n✅ 完成")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
