#!/usr/bin/env python
"""
复制渲染所需的最少必要文件到 models/{scene_id} 目录，并生成渲染配置文件

用法: 
    python script/setup_render_model.py --scene 031
    python script/setup_render_model.py --scene 031 --source output/waymo_full_exp/waymo_train_031

必需文件分析（基于 render.py 和 scene.py）:
1. 模型检查点: trained_model/iteration_N.pth
2. 相机数据: cameras.json (由 Dataset 加载)
3. 源数据路径: 需要在配置文件中指定 source_path
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def copy_minimal_files(scene_id, source_output_dir, target_models_dir):
    """
    复制渲染所需的最少必要文件
    
    Args:
        scene_id: 场景ID (如 "031")
        source_output_dir: 输出目录路径 (如 output/waymo_full_exp/waymo_train_031)
        target_models_dir: 目标目录 (如 models/031)
    
    Returns:
        dict: 包含复制信息的字典
    """
    
    print(f"📋 开始复制场景 {scene_id} 的渲染文件...")
    print(f"   源目录: {source_output_dir}")
    print(f"   目标目录: {target_models_dir}")
    print()
    
    # 创建目标目录
    os.makedirs(target_models_dir, exist_ok=True)
    
    files_copied = []
    total_size = 0
    
    # 1. 复制模型检查点 (.pth) - ⭐ 最关键
    trained_model_dir = os.path.join(source_output_dir, 'trained_model')
    if os.path.exists(trained_model_dir):
        pth_files = [f for f in os.listdir(trained_model_dir) if f.endswith('.pth')]
        if pth_files:
            target_trained_dir = os.path.join(target_models_dir, 'trained_model')
            os.makedirs(target_trained_dir, exist_ok=True)
            
            for pth_file in sorted(pth_files):
                src = os.path.join(trained_model_dir, pth_file)
                dst = os.path.join(target_trained_dir, pth_file)
                shutil.copy2(src, dst)
                size_mb = os.path.getsize(src) / (1024 * 1024)
                total_size += size_mb
                files_copied.append(('模型检查点', pth_file, size_mb))
                print(f"✅ 复制模型: {pth_file} ({size_mb:.2f} MB)")
        else:
            print("❌ 未找到 .pth 文件")
            return None
    else:
        print("❌ trained_model 目录不存在")
        return None
    
    # 2. 复制相机数据 (cameras.json) - ⭐ 必需
    cameras_json = os.path.join(source_output_dir, 'cameras.json')
    if os.path.exists(cameras_json):
        dst = os.path.join(target_models_dir, 'cameras.json')
        shutil.copy2(cameras_json, dst)
        size_kb = os.path.getsize(cameras_json) / 1024
        total_size += size_kb / 1024
        files_copied.append(('相机数据', 'cameras.json', size_kb / 1024))
        print(f"✅ 复制相机数据: cameras.json ({size_kb:.2f} KB)")
    else:
        print("⚠️  cameras.json 不存在（可能从 source_path 重新生成）")
    
    # 3. 复制配置文件 (cfg_args) - 用于参考
    cfg_args = os.path.join(source_output_dir, 'cfg_args')
    if os.path.exists(cfg_args):
        dst = os.path.join(target_models_dir, 'cfg_args')
        shutil.copy2(cfg_args, dst)
        files_copied.append(('配置', 'cfg_args', 0))
        print(f"✅ 复制配置: cfg_args")
    
    # 4. 复制输入点云文件 (input_ply/*.ply) - ⭐ 必需
    input_ply_dir = os.path.join(source_output_dir, 'input_ply')
    if os.path.exists(input_ply_dir):
        target_input_ply_dir = os.path.join(target_models_dir, 'input_ply')
        os.makedirs(target_input_ply_dir, exist_ok=True)
        
        ply_files = [f for f in os.listdir(input_ply_dir) if f.endswith('.ply')]
        for ply_file in sorted(ply_files):
            src = os.path.join(input_ply_dir, ply_file)
            dst = os.path.join(target_input_ply_dir, ply_file)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            total_size += size_mb
            files_copied.append(('输入点云', ply_file, size_mb))
            print(f"✅ 复制点云: {ply_file} ({size_mb:.2f} MB)")
    else:
        print("⚠️  input_ply 目录不存在")
    
    # 5. 复制天空贴图（可选，用于可视化）
    sky_latlong = os.path.join(source_output_dir, 'sky_latlong.png')
    if os.path.exists(sky_latlong):
        dst = os.path.join(target_models_dir, 'sky_latlong.png')
        shutil.copy2(sky_latlong, dst)
        size_kb = os.path.getsize(sky_latlong) / 1024
        total_size += size_kb / 1024
        files_copied.append(('天空贴图', 'sky_latlong.png', size_kb / 1024))
        print(f"✅ 复制天空贴图: sky_latlong.png ({size_kb:.2f} KB)")
    
    # 6. 复制最终点云文件（可选，用于可视化）
    point_cloud_dir = os.path.join(source_output_dir, 'point_cloud')
    if os.path.exists(point_cloud_dir):
        # 查找最新的迭代目录
        iter_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith('iteration_')]
        if iter_dirs:
            latest_iter = sorted(iter_dirs)[-1]  # 取最大的迭代号
            src_ply = os.path.join(point_cloud_dir, latest_iter, 'point_cloud.ply')
            if os.path.exists(src_ply):
                target_pc_dir = os.path.join(target_models_dir, 'point_cloud', latest_iter)
                os.makedirs(target_pc_dir, exist_ok=True)
                dst_ply = os.path.join(target_pc_dir, 'point_cloud.ply')
                shutil.copy2(src_ply, dst_ply)
                size_mb = os.path.getsize(src_ply) / (1024 * 1024)
                total_size += size_mb
                files_copied.append(('最终点云', f'{latest_iter}/point_cloud.ply', size_mb))
                print(f"✅ 复制最终点云: {latest_iter}/point_cloud.ply ({size_mb:.2f} MB)")
    
    print()
    print("=" * 70)
    print("📊 复制统计:")
    print(f"   文件数量: {len(files_copied)}")
    print(f"   总大小: {total_size:.2f} MB")
    print("=" * 70)
    
    return {
        'files_copied': files_copied,
        'total_size_mb': total_size,
        'target_dir': target_models_dir
    }


def create_render_config(scene_id, target_models_dir, source_data_path, output_dir=None):
    """
    创建渲染专用的配置文件
    
    Args:
        scene_id: 场景ID
        target_models_dir: 模型目录路径
        source_data_path: 原始数据路径 (source_path)
        output_dir: 输出目录（可选，默认使用 models/{scene_id}）
    
    Returns:
        str: 配置文件路径
    """
    
    if output_dir is None:
        output_dir = target_models_dir
    
    config_content = f"""# 渲染配置文件 - 场景 {scene_id}
# 自动生成于: {target_models_dir}

task: waymo_full_exp
source_path: {source_data_path}
exp_name: models_{scene_id}

# 模型路径配置
model_path: {target_models_dir}
point_cloud_dir: {os.path.join(target_models_dir, 'point_cloud')}
trained_model_dir: {os.path.join(target_models_dir, 'trained_model')}

data:
  split_test: -1
  split_train: -1
  type: Waymo
  white_background: false
  selected_frames: [0, 198]  # 根据实际数据调整
  cameras: [0, 1, 2]
  extent: 10
  use_colmap: true
  filter_colmap: true
  box_scale: 1.5

model:
  gaussian:
    sh_degree: 1
    fourier_dim: 5
    fourier_scale: 1.
    flip_prob: 0.
  nsg:
    include_bkgd: true
    include_obj: true
    include_sky: true
    opt_track: true
  sky:
    resolution: 1024
    white_background: false  # 添加天空背景配置

# 渲染模式：不需要训练参数
train:
  iterations: 1
  test_iterations: []
  save_iterations: []
  checkpoint_iterations: []

optim:
  # 渲染时不需要优化器参数，但需要占位
  prune_box_interval: 100
  densification_interval: 100

render:
  fps: 24
  concat_cameras: [1, 0, 2]
  save_image: true
  save_video: false

# 评估配置
eval:
  quiet: false
  skip_train: false
  skip_test: false

# 运行模式
mode: evaluate  # 或 'trajectory'
loaded_iter: -1  # -1 表示自动选择最新检查点
"""
    
    config_path = os.path.join(target_models_dir, f'render_{scene_id}.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ 创建配置文件: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description='复制渲染所需文件并生成配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python script/setup_render_model.py --scene 031
  
  # 指定源目录
  python script/setup_render_model.py --scene 031 --source output/waymo_full_exp/waymo_train_031
  
  # 指定数据路径
  python script/setup_render_model.py --scene 031 --data-path ./data/waymo/training/031
  
  # 自定义输出目录
  python script/setup_render_model.py --scene 031 --output models/custom_031
        """
    )
    
    parser.add_argument('--scene', type=str, required=True, help='场景ID (如: 031)')
    parser.add_argument('--source', type=str, default=None, 
                       help='训练输出目录 (默认: output/waymo_full_exp/waymo_train_{scene})')
    parser.add_argument('--data-path', type=str, default=None,
                       help='原始数据路径 (默认: ./data/waymo/training/{scene})')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录 (默认: models/{scene})')
    
    args = parser.parse_args()
    
    # 设置默认路径
    scene_id = args.scene
    if args.source is None:
        source_output_dir = f'output/waymo_full_exp/waymo_train_{scene_id}'
    else:
        source_output_dir = args.source
    
    if args.data_path is None:
        source_data_path = f'./data/waymo/training/{scene_id}'
    else:
        source_data_path = args.data_path
    
    if args.output is None:
        target_models_dir = f'models/{scene_id}'
    else:
        target_models_dir = args.output
    
    # 检查源目录是否存在
    if not os.path.exists(source_output_dir):
        print(f"❌ 错误: 源目录不存在 - {source_output_dir}")
        print(f"   请确认训练已完成，或手动指定 --source 参数")
        sys.exit(1)
    
    # 复制文件
    result = copy_minimal_files(scene_id, source_output_dir, target_models_dir)
    
    if result is None:
        print("\n❌ 文件复制失败")
        sys.exit(1)
    
    # 创建配置文件
    print()
    config_path = create_render_config(scene_id, target_models_dir, source_data_path)
    
    # 打印使用说明
    print()
    print("=" * 70)
    print("🎉 设置完成！")
    print("=" * 70)
    print()
    print("📁 目录结构:")
    print(f"   {target_models_dir}/")
    print(f"   ├── trained_model/")
    print(f"   │   └── iteration_50000.pth")
    print(f"   ├── cameras.json")
    print(f"   ├── sky_latlong.png (可选)")
    print(f"   └── render_{scene_id}.yaml")
    print()
    print("🚀 渲染命令:")
    print(f"   python render.py --config {config_path} mode evaluate")
    print()
    print("🎬 轨迹渲染:")
    print(f"   python render.py --config {config_path} mode trajectory")
    print()
    print("⚠️  重要提示:")
    print(f"   1. 确保 source_path 指向正确的数据目录:")
    print(f"      {source_data_path}")
    print(f"   2. 数据目录应包含:")
    print(f"      - images/*.png")
    print(f"      - intrinsics/*.txt")
    print(f"      - extrinsics/*.txt")
    print(f"      - timestamps.json")
    print(f"      - track/track_info.txt")
    print(f"   3. 如果数据路径不同，请编辑 {config_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
