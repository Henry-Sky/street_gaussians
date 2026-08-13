import torch
import os
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.datasets.dataset import Dataset
from lib.config import cfg
from lib.utils.general_utils import safe_state

def print_model_structure():
    """加载并打印StreetGaussianModel的完整结构"""
    
    print("=" * 80)
    print("开始加载模型...")
    print("=" * 80)
    
    # 初始化数据集以获取metadata
    dataset = Dataset()
    
    # 创建模型实例
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    # 尝试加载训练好的模型
    model_path = cfg.model_path
    trained_model_dir = cfg.get('trained_model_dir', os.path.join(model_path, 'point_cloud'))
    
    # 查找最新的checkpoint
    if cfg.loaded_iter == -1:
        # 自动查找最大迭代次数
        from lib.utils.system_utils import searchForMaxIteration
        loaded_iter = searchForMaxIteration(trained_model_dir)
    else:
        loaded_iter = cfg.loaded_iter
    
    if loaded_iter is not None:
        ckpt_path = os.path.join(trained_model_dir, f'iteration_{loaded_iter}.pth')
        if os.path.exists(ckpt_path):
            print(f"\n找到模型文件: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            print(f"成功加载第 {loaded_iter} 次迭代的模型")
            
            # 加载到模型中
            gaussians.load_state_dict(state_dict)
        else:
            print(f"\n警告: 未找到模型文件 {ckpt_path}")
            print("将仅显示模型架构（未加载权重）")
    else:
        print("\n警告: 未找到任何训练好的模型")
        print("将仅显示模型架构（未加载权重）")
    
    print("\n" + "=" * 80)
    print("模型结构信息")
    print("=" * 80)
    
    # 1. 打印基本配置信息
    print("\n【基本配置】")
    print(f"  - 包含背景 (include_background): {gaussians.include_background}")
    print(f"  - 包含动态对象 (include_obj): {gaussians.include_obj}")
    print(f"  - 包含天空 (include_sky): {gaussians.include_sky}")
    print(f"  - 最大SH阶数 (max_sh_degree): {gaussians.max_sh_degree}")
    print(f"  - Fourier维度 (fourier_dim): {gaussians.fourier_dim}")
    print(f"  - 使用颜色校正 (use_color_correction): {gaussians.use_color_correction}")
    print(f"  - 使用位姿校正 (use_pose_correction): {gaussians.use_pose_correction}")
    
    # 2. 打印模型组件数量
    print("\n【模型组件统计】")
    print(f"  - 总模型数量 (models_num): {gaussians.models_num}")
    print(f"  - 模型名称映射 (model_name_id):")
    for model_name, model_id in gaussians.model_name_id.items():
        print(f"    * {model_name}: ID={model_id}")
    
    # 3. 打印各子模型详细信息
    print("\n【子模型详细信息】")
    
    # 背景模型
    if hasattr(gaussians, 'background'):
        print("\n  [背景模型 - background]")
        bg_model = gaussians.background
        print(f"    - 高斯点数量: {bg_model.get_xyz.shape[0] if hasattr(bg_model, 'get_xyz') else 'N/A'}")
        print(f"    - SH特征维度: {bg_model.get_features.shape[1:] if hasattr(bg_model, 'get_features') else 'N/A'}")
        if hasattr(bg_model, 'optimizer'):
            print(f"    - 优化器参数数量: {sum(p.numel() for p in bg_model.optimizer.param_groups[0]['params'])}")
    
    # 动态对象模型
    if hasattr(gaussians, 'obj_list') and gaussians.obj_list:
        print(f"\n  [动态对象模型 - 共{len(gaussians.obj_list)}个对象]")
        for obj_name in gaussians.obj_list[:5]:  # 只显示前5个
            obj_model = getattr(gaussians, obj_name)
            num_points = obj_model.get_xyz.shape[0] if hasattr(obj_model, 'get_xyz') else 'N/A'
            print(f"    - {obj_name}: {num_points} 个高斯点")
        if len(gaussians.obj_list) > 5:
            print(f"    ... 还有 {len(gaussians.obj_list) - 5} 个对象")
    
    # 天空模型
    if hasattr(gaussians, 'sky_cubemap') and gaussians.sky_cubemap is not None:
        print("\n  [天空立方体贴图 - sky_cubemap]")
        sky_model = gaussians.sky_cubemap
        param_count = sum(p.numel() for p in sky_model.parameters())
        print(f"    - 参数量: {param_count:,}")
    
    # Actor姿态网络
    if hasattr(gaussians, 'actor_pose') and gaussians.actor_pose is not None:
        print("\n  [Actor姿态网络 - actor_pose]")
        actor_model = gaussians.actor_pose
        param_count = sum(p.numel() for p in actor_model.parameters())
        print(f"    - 参数量: {param_count:,}")
        print(f"    - 网络结构:")
        for name, module in actor_model.named_modules():
            if isinstance(module, torch.nn.Module) and len(list(module.children())) == 0:
                indent = "      " + "  " * (name.count('.') + 1)
                print(f"{indent}{type(module).__name__}: {module}")
    
    # 颜色校正网络
    if hasattr(gaussians, 'color_correction') and gaussians.color_correction is not None:
        print("\n  [颜色校正网络 - color_correction]")
        color_model = gaussians.color_correction
        param_count = sum(p.numel() for p in color_model.parameters())
        print(f"    - 参数量: {param_count:,}")
    
    # 位姿校正网络
    if hasattr(gaussians, 'pose_correction') and gaussians.pose_correction is not None:
        print("\n  [位姿校正网络 - pose_correction]")
        pose_model = gaussians.pose_correction
        param_count = sum(p.numel() for p in pose_model.parameters())
        print(f"    - 参数量: {param_count:,}")
    
    # 4. 打印总体统计
    print("\n【总体统计】")
    total_params = sum(p.numel() for p in gaussians.parameters())
    trainable_params = sum(p.numel() for p in gaussians.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    
    # 计算高斯点总数
    total_gaussians = 0
    if hasattr(gaussians, 'background'):
        total_gaussians += gaussians.background.get_xyz.shape[0]
    if hasattr(gaussians, 'obj_list'):
        for obj_name in gaussians.obj_list:
            obj_model = getattr(gaussians, obj_name)
            total_gaussians += obj_model.get_xyz.shape[0]
    if hasattr(gaussians, 'sky') and gaussians.sky is not None:
        total_gaussians += gaussians.sky.get_xyz.shape[0]
    
    print(f"  - 高斯点总数: {total_gaussians:,}")
    
    # 5. 打印完整PyTorch模块结构（可选，详细模式）
    if cfg.get('verbose', False):
        print("\n【完整PyTorch模块树】")
        print(gaussians)
    
    print("\n" + "=" * 80)
    print("模型结构展示完成")
    print("=" * 80)

if __name__ == "__main__":
    print("测试模式 - 加载并分析模型结构")
    safe_state(cfg.eval.quiet)
    print_model_structure()
