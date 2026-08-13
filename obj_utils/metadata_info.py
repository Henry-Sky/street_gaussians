# obj_info.py

import os
import sys
import numpy as np
import torch
from lib.config import cfg
from lib.datasets.dataset import sceneLoadTypeCallbacks


def print_camera_info(scene_info):
    """打印相机信息"""
    print("=" * 60)
    print("相机信息 (Cameras)")
    print("=" * 60)
    
    train_cams = scene_info.train_cameras
    test_cams = scene_info.test_cameras
    
    print(f"训练集相机数: {len(train_cams)}")
    print(f"测试集相机数: {len(test_cams)}")
    print(f"相机总数: {len(train_cams) + len(test_cams)}\n")
    
    if len(train_cams) > 0:
        # 显示前3个训练相机的详细信息
        print("训练集相机示例 (前3个):")
        for i, cam in enumerate(train_cams[:3]):
            print(f"\n相机 {i+1}:")
            print(f"  UID: {cam.uid}")
            print(f"  图像名称: {cam.image_name}")
            print(f"  图像路径: {cam.image_path}")
            print(f"  分辨率: {cam.width} x {cam.height}")
            print(f"  FovX: {cam.FovX:.4f}, FovY: {cam.FovY:.4f}")
            print(f"  内参矩阵 K:\n{cam.K}")
            print(f"  旋转矩阵 R:\n{cam.R}")
            print(f"  平移向量 T: {cam.T}")
            
            # 从 metadata 中提取额外信息
            if hasattr(cam, 'metadata') and cam.metadata:
                meta = cam.metadata
                print(f"  帧索引: {meta.get('frame_idx', 'N/A')}")
                print(f"  相机ID: {meta.get('cam', 'N/A')}")
                print(f"  时间戳: {meta.get('timestamp', 'N/A')}")
    
    print()


def print_metadata_info(scene_info):
    """打印场景元数据信息"""
    print("=" * 60)
    print("场景元数据 (Scene Metadata)")
    print("=" * 60)
    
    metadata = scene_info.metadata
    
    if metadata:
        print(f"图像总数: {metadata.get('num_images', 'N/A')}")
        print(f"相机数量: {metadata.get('num_cams', 'N/A')}")
        print(f"帧数: {metadata.get('num_frames', 'N/A')}")
        
        # 动态对象信息
        if 'obj_tracklets' in metadata:
            obj_tracklets = metadata['obj_tracklets']
            print(f"\n动态对象轨迹数: {len(obj_tracklets)}")
            if len(obj_tracklets) > 0:
                # 检查 obj_tracklets 的类型
                if isinstance(obj_tracklets, dict):
                    print(f"  对象ID列表: {list(obj_tracklets.keys())[:10]}{'...' if len(obj_tracklets) > 10 else ''}")
                elif isinstance(obj_tracklets, np.ndarray):
                    # 如果是数组，显示形状和基本信息
                    print(f"  数据结构: shape={obj_tracklets.shape}, dtype={obj_tracklets.dtype}")
                    if obj_tracklets.ndim == 3:
                        print(f"  维度含义: [帧数={obj_tracklets.shape[0]}, 每帧最大对象数={obj_tracklets.shape[1]}, 特征维度={obj_tracklets.shape[2]}]")
                        # 统计每帧实际的对象数量（排除-1填充）
                        valid_counts = []
                        for frame_data in obj_tracklets:
                            # 假设第一列是对象ID，-1表示无效
                            valid_objs = np.sum(frame_data[:, 0] >= 0)
                            valid_counts.append(valid_objs)
                        print(f"  每帧对象数: min={min(valid_counts)}, max={max(valid_counts)}, avg={np.mean(valid_counts):.1f}")
                    else:
                        sample_ids = obj_tracklets[:10] if len(obj_tracklets) > 10 else obj_tracklets
                        print(f"  对象ID示例: {sample_ids}{'...' if len(obj_tracklets) > 10 else ''}")
                else:
                    print(f"  对象ID类型: {type(obj_tracklets).__name__}")
        
        if 'obj_meta' in metadata:
            obj_meta = metadata['obj_meta']
            print(f"\n对象元数据条目数: {len(obj_meta)}")
        
        if 'tracklet_timestamps' in metadata:
            tracklet_ts = metadata['tracklet_timestamps']
            print(f"\n轨迹时间戳数: {len(tracklet_ts)}")
    
    print()


def print_point_cloud_info(scene_info):
    """打印点云信息"""
    print("=" * 60)
    print("点云信息 (Point Cloud)")
    print("=" * 60)
    
    point_cloud = scene_info.point_cloud
    
    if point_cloud is not None:
        points = point_cloud.points
        colors = point_cloud.colors
        normals = point_cloud.normals
        
        print(f"点数: {len(points)}")
        
        if len(points) > 0:
            print("\n坐标范围:")
            print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            print("\n边界框尺寸:")
            bbox_size = points.max(axis=0) - points.min(axis=0)
            print(f"  X: {bbox_size[0]:.2f}")
            print(f"  Y: {bbox_size[1]:.2f}")
            print(f"  Z: {bbox_size[2]:.2f}")
            print(f"  对角线长度: {np.linalg.norm(bbox_size):.2f}")
            
            print("\n中心点坐标:")
            center = points.mean(axis=0)
            print(f"  ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            
            print("\n颜色值范围 (RGB):")
            print(f"  R: [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
            print(f"  G: [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
            print(f"  B: [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
            
            if normals is not None and len(normals) > 0:
                print("\n法向量统计:")
                print(f"  法向量均值: ({normals[:, 0].mean():.4f}, {normals[:, 1].mean():.4f}, {normals[:, 2].mean():.4f})")
    else:
        print("没有点云数据")
    
    print()


def print_normalization_info(scene_info):
    """打印归一化信息"""
    print("=" * 60)
    print("归一化信息 (Normalization)")
    print("=" * 60)
    
    nerf_norm = scene_info.nerf_normalization
    
    if nerf_norm:
        print(f"平移向量: {nerf_norm.get('translate', 'N/A')}")
        print(f"半径: {nerf_norm.get('radius', 'N/A'):.4f}")
        if 'center' in nerf_norm:
            center = nerf_norm['center']
            print(f"中心点: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    print()


def print_summary(scene_info):
    """打印数据集摘要"""
    print("\n" + "=" * 60)
    print("数据集摘要 (Dataset Summary)")
    print("=" * 60)
    
    train_cams = scene_info.train_cameras
    test_cams = scene_info.test_cameras
    point_cloud = scene_info.point_cloud
    metadata = scene_info.metadata
    
    print(f"训练相机数:   {len(train_cams)}")
    print(f"测试相机数:   {len(test_cams)}")
    print(f"相机总数:     {len(train_cams) + len(test_cams)}")
    
    if point_cloud is not None:
        print(f"点云数量:     {len(point_cloud.points)}")
    
    if metadata:
        print(f"图像总数:     {metadata.get('num_images', 'N/A')}")
        print(f"帧数:         {metadata.get('num_frames', 'N/A')}")
        if 'obj_tracklets' in metadata:
            print(f"动态对象数:   {len(metadata['obj_tracklets'])}")
    
    print("=" * 60)


def print_obj_metadata_detailed(scene_info):
    """打印详细的对象元数据信息"""
    print("=" * 60)
    print("动态对象详细元数据 (Object Metadata Details)")
    print("=" * 60)
    
    metadata = scene_info.metadata
    if 'obj_meta' not in metadata:
        print("没有对象元数据")
        return
    
    obj_meta = metadata['obj_meta']
    
    # 检查类型
    if isinstance(obj_meta, dict):
        print(f"\n对象总数: {len(obj_meta)}\n")
        
        for track_id, obj_info in obj_meta.items():
            print(f"{'─' * 60}")
            print(f"对象 ID: {track_id}")
            print(f"{'─' * 60}")
            
            # 基本信息
            print(f"  类别:           {obj_info.get('class', 'N/A')}")
            print(f"  类别标签:       {obj_info.get('class_label', 'N/A')}")
            print(f"  可变形:         {'是' if obj_info.get('deformable', False) else '否'}")
            
            # 尺寸信息
            print(f"\n  尺寸 (米):")
            print(f"    长度:         {obj_info.get('length', 'N/A'):.3f}")
            print(f"    宽度:         {obj_info.get('width', 'N/A'):.3f}")
            print(f"    高度:         {obj_info.get('height', 'N/A'):.3f}")
            
            # 体积估算
            if all(k in obj_info for k in ['length', 'width', 'height']):
                volume = obj_info['length'] * obj_info['width'] * obj_info['height']
                print(f"    估算体积:     {volume:.3f} m³")
            
            # 时间信息
            print(f"\n  时间范围:")
            if 'start_frame' in obj_info and 'end_frame' in obj_info:
                duration_frames = obj_info['end_frame'] - obj_info['start_frame'] + 1
                print(f"    起始帧:       {obj_info['start_frame']}")
                print(f"    结束帧:       {obj_info['end_frame']}")
                print(f"    持续帧数:     {duration_frames}")
            
            if 'start_timestamp' in obj_info and 'end_timestamp' in obj_info:
                duration_time = obj_info['end_timestamp'] - obj_info['start_timestamp']
                print(f"    起始时间:     {obj_info['start_timestamp']:.4f}s")
                print(f"    结束时间:     {obj_info['end_timestamp']:.4f}s")
                print(f"    持续时间:     {duration_time:.4f}s")
            
            print()
    
    elif isinstance(obj_meta, np.ndarray):
        print(f"\n对象元数据为数组格式，shape: {obj_meta.shape}")
        print("提示: 此数据集使用数组格式存储对象元数据")
    
    else:
        print(f"\n对象元数据类型: {type(obj_meta).__name__}")
    
    print("=" * 60)


def save_obj_metadata_separately(scene_info, output_dir=None):
    """
    将每个对象的元数据单独保存为JSON文件
    
    Args:
        scene_info: 场景信息对象
        output_dir: 输出目录，默认为 model_path/obj_metadata
    """
    import json
    from lib.config import cfg
    
    metadata = scene_info.metadata
    if 'obj_meta' not in metadata:
        print("警告: 没有对象元数据可保存")
        return []
    
    obj_meta = metadata['obj_meta']
    
    # 只支持字典格式
    if not isinstance(obj_meta, dict):
        print(f"警告: obj_meta 类型为 {type(obj_meta).__name__}，不支持单独保存")
        return []
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(cfg.model_path, 'obj_metadata')
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for track_id, obj_info in obj_meta.items():
        # 构建JSON文件路径
        json_path = os.path.join(output_dir, f'obj_{int(track_id):03d}.json')
        
        # 确保所有值都是JSON可序列化的
        serializable_info = {}
        for key, value in obj_info.items():
            if isinstance(value, (np.integer,)):
                serializable_info[key] = int(value)
            elif isinstance(value, (np.floating,)):
                serializable_info[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_info[key] = value.tolist()
            else:
                serializable_info[key] = value
        
        # 保存为JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2, ensure_ascii=False)
        
        saved_files.append(json_path)
        print(f"✓ 保存对象 {track_id}: {json_path}")
    
    print(f"\n共保存 {len(saved_files)} 个对象元数据文件到: {output_dir}")
    return saved_files


def print_obj_tracklets_info(scene_info):
    """打印对象轨迹信息（位置、朝向、轨迹）"""
    print("=" * 60)
    print("动态对象轨迹信息 (Object Tracklets - Position & Orientation)")
    print("=" * 60)
    
    metadata = scene_info.metadata
    
    if 'obj_tracklets' not in metadata:
        print("没有对象轨迹数据")
        return
    
    obj_tracklets = metadata['obj_tracklets']
    
    # 检查类型
    if isinstance(obj_tracklets, np.ndarray):
        print(f"\n数据结构: shape={obj_tracklets.shape}")
        print(f"数据类型: {obj_tracklets.dtype}")
        
        if obj_tracklets.ndim == 3:
            num_frames, max_obj_per_frame, num_features = obj_tracklets.shape
            print(f"\n维度含义:")
            print(f"  - 帧数: {num_frames}")
            print(f"  - 每帧最大对象数: {max_obj_per_frame}")
            print(f"  - 特征维度: {num_features}")
            
            if num_features >= 8:
                print(f"\n特征维度解析 ([track_id, x, y, z, qw, qx, qy, qz]):")
                print(f"  [0] track_id: 对象ID (-1表示无效)")
                print(f"  [1-3] position: 位置坐标 (x, y, z)")
                print(f"  [4-7] orientation: 朝向四元数 (qw, qx, qy, qz)")
                
                # 统计有效对象
                valid_mask = obj_tracklets[:, :, 0] >= 0
                total_valid = np.sum(valid_mask)
                print(f"\n统计信息:")
                print(f"  - 总有效观测数: {total_valid}")
                print(f"  - 平均每帧对象数: {np.mean(np.sum(valid_mask, axis=1)):.2f}")
                print(f"  - 最多对象帧: {np.max(np.sum(valid_mask, axis=1))}")
                
                # 获取所有唯一的track_id
                unique_ids = np.unique(obj_tracklets[:,:,0][valid_mask])
                print(f"  - 唯一对象ID数: {len(unique_ids)}")
                print(f"  - 对象ID列表: {unique_ids[:20]}{'...' if len(unique_ids) > 20 else ''}")
                
                # 显示前几个对象的轨迹示例
                print(f"\n轨迹示例 (前3个对象):")
                for i, track_id in enumerate(unique_ids[:3]):
                    print(f"\n  对象 ID {int(track_id)}:")
                    # 找到该对象的所有帧
                    obj_mask = (obj_tracklets[:, :, 0] == track_id)
                    frame_indices = np.where(np.any(obj_mask, axis=1))[0]
                    
                    if len(frame_indices) > 0:
                        print(f"    出现帧数: {len(frame_indices)} 帧")
                        print(f"    帧范围: [{frame_indices[0]}, {frame_indices[-1]}]")
                        
                        # 显示前3帧的位置和朝向
                        for j, frame_idx in enumerate(frame_indices[:3]):
                            obj_col = np.where(obj_mask[frame_idx])[0][0]
                            pos = obj_tracklets[frame_idx, obj_col, 1:4]
                            quat = obj_tracklets[frame_idx, obj_col, 4:8]
                            
                            # 计算欧拉角（简化版，绕Z轴的偏航角）
                            yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                                           1-2*(quat[2]**2 + quat[3]**2))
                            yaw_deg = np.degrees(yaw)
                            
                            print(f"      帧 {frame_idx:3d}: 位置=({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}), "
                                  f"偏航角={yaw_deg:7.1f}°")
                        
                        if len(frame_indices) > 3:
                            print(f"      ... 还有 {len(frame_indices)-3} 帧")
                            
                        # 计算轨迹长度
                        if len(frame_indices) > 1:
                            first_pos = None
                            last_pos = None
                            for frame_idx in [frame_indices[0], frame_indices[-1]]:
                                obj_col = np.where(obj_mask[frame_idx])[0][0]
                                pos = obj_tracklets[frame_idx, obj_col, 1:4]
                                if first_pos is None:
                                    first_pos = pos
                                last_pos = pos
                            
                            distance = np.linalg.norm(last_pos - first_pos)
                            print(f"    移动距离: {distance:.2f} 米")
    
    elif isinstance(obj_tracklets, dict):
        print(f"\n对象轨迹为字典格式，包含 {len(obj_tracklets)} 个对象")
        for track_id, trajectory in list(obj_tracklets.items())[:3]:
            print(f"\n  对象 {track_id}:")
            if isinstance(trajectory, np.ndarray):
                print(f"    轨迹shape: {trajectory.shape}")
                print(f"    帧数: {len(trajectory)}")
    
    else:
        print(f"\n对象轨迹类型: {type(obj_tracklets).__name__}")
    
    print("=" * 60)


def save_obj_trajectories(scene_info, output_dir=None):
    """
    保存每个对象的完整轨迹（位置+朝向）为JSON文件
    
    Args:
        scene_info: 场景信息对象
        output_dir: 输出目录，默认为 model_path/obj_trajectories
    """
    import json
    from lib.config import cfg
    
    metadata = scene_info.metadata
    
    if 'obj_tracklets' not in metadata or 'obj_meta' not in metadata:
        print("警告: 缺少轨迹或元数据")
        return []
    
    obj_tracklets = metadata['obj_tracklets']
    obj_meta = metadata['obj_meta']
    
    # 只支持数组格式的tracklets
    if not isinstance(obj_tracklets, np.ndarray) or obj_tracklets.ndim != 3:
        print(f"警告: obj_tracklets 格式不支持，shape={obj_tracklets.shape if isinstance(obj_tracklets, np.ndarray) else 'N/A'}")
        return []
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(cfg.model_path, 'obj_trajectories')
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames, max_obj_per_frame, num_features = obj_tracklets.shape
    
    # 获取所有唯一的track_id
    valid_mask = obj_tracklets[:, :, 0] >= 0
    unique_ids = np.unique(obj_tracklets[:,:,0][valid_mask])
    
    saved_files = []
    
    for track_id in unique_ids:
        track_id_int = int(track_id)
        
        # 构建轨迹数据
        trajectory = []
        
        for frame_idx in range(num_frames):
            # 查找该帧中此对象的列
            obj_mask = (obj_tracklets[frame_idx, :, 0] == track_id)
            if not np.any(obj_mask):
                continue
            
            obj_col = np.where(obj_mask)[0][0]
            
            # 提取位置和朝向
            position = obj_tracklets[frame_idx, obj_col, 1:4].tolist()
            quaternion = obj_tracklets[frame_idx, obj_col, 4:8].tolist()
            
            # 计算偏航角（yaw）
            qw, qx, qy, qz = quaternion
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))
            
            frame_data = {
                "frame": frame_idx,
                "position": {
                    "x": round(position[0], 4),
                    "y": round(position[1], 4),
                    "z": round(position[2], 4)
                },
                "orientation": {
                    "quaternion": [round(q, 6) for q in quaternion],
                    "yaw_rad": round(yaw, 6),
                    "yaw_deg": round(np.degrees(yaw), 2)
                }
            }
            
            trajectory.append(frame_data)
        
        if len(trajectory) == 0:
            continue
        
        # 添加元数据
        obj_info = obj_meta.get(track_id_int, {})
        
        trajectory_data = {
            "track_id": track_id_int,
            "class": obj_info.get('class', 'unknown'),
            "deformable": obj_info.get('deformable', False),
            "dimensions": {
                "length": obj_info.get('length', 0),
                "width": obj_info.get('width', 0),
                "height": obj_info.get('height', 0)
            },
            "time_range": {
                "start_frame": trajectory[0]['frame'],
                "end_frame": trajectory[-1]['frame'],
                "duration_frames": len(trajectory)
            },
            "trajectory": trajectory
        }
        
        # 保存为JSON
        json_path = os.path.join(output_dir, f'traj_{track_id_int:03d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        saved_files.append(json_path)
        print(f"✓ 保存对象 {track_id_int} 轨迹 ({len(trajectory)} 帧): {json_path}")
    
    print(f"\n共保存 {len(saved_files)} 个对象轨迹文件到: {output_dir}")
    return saved_files


def main():
    """主函数"""
    # 检查必要配置
    if not cfg.source_path:
        print("错误: 未配置 source_path")
        print("\n用法:")
        print("  python test.py --config <config_file>")
        print("\n示例:")
        print("  python test.py --config configs/example/waymo_train_031.yaml")
        sys.exit(1)
    
    print(f"数据源路径: {cfg.source_path}")
    print(f"数据类型: {cfg.data.get('type', 'Colmap')}")
    print(f"模式: {cfg.mode}\n")
    
    try:
        # 直接调用场景读取函数，避免Dataset类自动加载相机图像（节省内存）
        print("正在加载数据集元信息...")
        dataset_type = cfg.data.get('type', "Colmap")
        assert dataset_type in sceneLoadTypeCallbacks.keys(), f'Could not recognize scene type: {dataset_type}'
        
        # 直接读取场景信息，不加载实际图像数据
        scene_info = sceneLoadTypeCallbacks[dataset_type](cfg.source_path, **cfg.data)
        
        # 打印信息
        print_camera_info(scene_info)
        print_metadata_info(scene_info)
        print_obj_metadata_detailed(scene_info)  # 对象静态元数据（尺寸、类别等）
        print_obj_tracklets_info(scene_info)      # 对象动态轨迹（位置、朝向）
        print_point_cloud_info(scene_info)
        print_normalization_info(scene_info)
        print_summary(scene_info)
        
        # 可选：保存每个对象的元数据为单独的JSON文件
        save_obj_metadata_separately(scene_info)
        
        # 可选：保存每个对象的完整轨迹
        save_obj_trajectories(scene_info)
        
    except Exception as e:
        print(f"\n错误: 加载数据集时出错")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
