#!/usr/bin/env python
"""
简单查询 .pth 模型参数结构
用法: python script/query_pth.py path/to/model.pth
"""

import torch
import sys
import os


def format_size(num_bytes):
    """将字节转换为人类可读的大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def query_pth(checkpoint_path):
    """查询 pth 文件结构"""
    
    # 检查文件
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 文件不存在 - {checkpoint_path}")
        return False
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"📁 文件: {os.path.basename(checkpoint_path)}")
    print(f"📊 大小: {format_size(file_size)}")
    print("=" * 70)
    
    # 加载模型
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 显示顶层信息
    print("\n🔑 顶层组件:")
    for key in state_dict.keys():
        if key == 'iter':
            print(f"   • {key}: {state_dict[key]} (训练迭代次数)")
        else:
            print(f"   • {key}")
    
    print("\n" + "=" * 70)
    print("📊 组件详情:\n")
    
    total_memory = 0
    total_points = 0
    
    # 遍历每个组件
    for key, value in state_dict.items():
        if key == 'iter':
            continue
        
        print(f"【{key}】")
        
        if isinstance(value, dict):
            # 高斯模型 (background, obj_XXX)
            if 'xyz' in value:
                num_points = value['xyz'].shape[0]
                total_points += num_points
                
                # 计算内存
                comp_memory = 0
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        comp_memory += v.element_size() * v.numel()
                
                total_memory += comp_memory
                
                print(f"   类型: GaussianModel")
                print(f"   高斯点数: {num_points:,}")
                
                # 显示主要参数形状
                if 'feature_dc' in value:
                    print(f"   feature_dc: {tuple(value['feature_dc'].shape)}")
                if 'feature_rest' in value:
                    print(f"   feature_rest: {tuple(value['feature_rest'].shape)}")
                if 'active_sh_degree' in value:
                    print(f"   SH degree: {value['active_sh_degree']}")
                
                has_optimizer = 'optimizer' in value
                print(f"   包含优化器: {'✅' if has_optimizer else '❌'}")
                print(f"   内存占用: {format_size(comp_memory)}")
            
            # 模块 (actor_pose, sky_cubemap, etc.)
            elif 'params' in value:
                params = value['params']
                
                comp_memory = 0
                for k, v in params.items():
                    if isinstance(v, torch.Tensor):
                        comp_memory += v.element_size() * v.numel()
                
                total_memory += comp_memory
                
                print(f"   类型: Module")
                print(f"   参数张量数: {len(params)}")
                
                # 显示主要参数
                for param_name, param_tensor in params.items():
                    if isinstance(param_tensor, torch.Tensor):
                        print(f"   {param_name}: {tuple(param_tensor.shape)}")
                
                has_optimizer = 'optimizer' in value
                print(f"   包含优化器: {'✅' if has_optimizer else '❌'}")
                print(f"   内存占用: {format_size(comp_memory)}")
        
        print()  # 空行分隔
    
    # 总结
    print("=" * 70)
    print(f"📈 统计信息:")
    print(f"   总高斯点数: {total_points:,}")
    print(f"   预估内存: {format_size(total_memory)}")
    print(f"   磁盘大小: {format_size(file_size)}")
    print(f"    overhead: {format_size(file_size - total_memory)}")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python script/query_pth.py <checkpoint_path>")
        print("示例: python script/query_pth.py output/scene/trained_model/iteration_30000.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = query_pth(checkpoint_path)
    sys.exit(0 if success else 1)
