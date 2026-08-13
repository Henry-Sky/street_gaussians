# StreetGaussians 动态对象解耦快速参考

> **核心原则**: 每个动态对象独立存储，支持单独提取/编辑/替换

---

## 📦 数据结构

### .pth文件组织
```python
{
    'background': {xyz, features, scaling, rotation, ...},
    'obj_001': {xyz, features, scaling, rotation, ...},  # 独立存储
    'obj_002': {xyz, features, scaling, rotation, ...},  # 独立存储
    'actor_pose': {opt_trans, opt_rots},  # 统一位姿管理
}
```

**关键特性**:
- ✅ 每个对象 = 独立 `GaussianModelActor` 实例
- ✅ 局部坐标系存储（相对对象中心）
- ✅ 通过 `track_id` 索引查询位姿

---

## 🔧 核心操作

### 1️⃣ 提取对象

```python
import torch
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.datasets.dataset import Dataset

def extract_object(model_path, track_id, output_path):
    """提取单个对象为独立.pth文件"""
    cfg.model_path = model_path
    cfg.mode = 'evaluate'
    
    # 加载模型
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    from lib.utils.system_utils import searchForMaxIteration
    ckpt_path = f"{model_path}/point_cloud/iteration_{searchForMaxIteration(...)}.pth"
    gaussians.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    
    # 提取对象
    obj_name = f'obj_{track_id:03d}'
    obj_model = getattr(gaussians, obj_name)
    
    obj_state = {
        'gaussian_params': obj_model.state_dict(is_final=True),
        'obj_meta': gaussians.obj_info[track_id],
        'fourier_config': {
            'fourier_dim': obj_model.fourier_dim,
            'fourier_scale': obj_model.fourier_scale,
        }
    }
    
    torch.save(obj_state, output_path)
    print(f"✅ 提取完成: {output_path}")
```

### 2️⃣ 导出PLY（外部编辑）

```python
def export_to_ply(model_path, track_id, output_dir):
    """导出为PLY格式用于Blender等软件编辑"""
    # ... 加载模型同上 ...
    
    obj_model = getattr(gaussians, f'obj_{track_id:03d}')
    ply_path = f"{output_dir}/obj_{track_id:03d}.ply"
    obj_model.save_ply(ply_path)
    
    # 保存元数据
    import json
    with open(f"{output_dir}/obj_{track_id:03d}_meta.json", 'w') as f:
        json.dump(gaussians.obj_info[track_id], f, default=str)
```

### 3️⃣ 替换对象

```python
def replace_object(model_path, track_id, edited_obj_path, output_path=None):
    """用编辑后的对象替换原模型中的对象"""
    # 加载原模型
    cfg.model_path = model_path
    cfg.mode = 'evaluate'
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    # ... 加载checkpoint ...
    
    # 加载编辑后的对象
    edited_state = torch.load(edited_obj_path, map_location='cuda')
    obj_model = getattr(gaussians, f'obj_{track_id:03d}')
    
    # 替换参数
    obj_model.load_state_dict(edited_state['gaussian_params'])
    
    # 可选：更新位姿
    if 'pose_trajectory' in edited_state:
        update_pose(gaussians.actor_pose, track_id, edited_state['pose_trajectory'])
    
    # 保存
    if output_path is None:
        output_path = model_path.replace('.pth', '_modified.pth')
    torch.save(gaussians.save_state_dict(is_final=True), output_path)
```

---

## ⚠️ 关键注意事项

### 位姿依赖
- 对象高斯点在**局部坐标系**
- 渲染时需 `ActorPose` 提供全局变换
- 提取时建议包含 `pose_trajectory`

### Fourier SH时间维度
```python
# 外观随时间变化，编辑时保持维度一致
features_dc shape: [N, 3, fourier_dim]
```

### 编辑建议
1. **备份原模型** 🔒
2. **小幅度修改** → 直接编辑张量
3. **大幅度修改** → PLY外部编辑后重载
4. **验证渲染** → 替换后测试效果
5. **微调优化** → 重大修改后100-500步微调

---

## 🛠️ 常用工具脚本

### extract_objects.py
```bash
# 提取单个对象
python extract_objects.py --model_path output/xxx --track_id 5 --format pth

# 提取所有对象为PLY
python extract_objects.py --model_path output/xxx --all --format ply
```

### replace_object.py
```bash
python replace_object.py \
    --model_path output/waymo_val_006 \
    --track_id 5 \
    --obj_path obj_005_edited.pth \
    --output_path output/modified.pth
```

---

## 📊 应用场景速查

| 场景 | 方法 | 难度 |
|------|------|------|
| 删除对象 | 从state_dict移除key | ⭐ |
| 修改颜色 | 编辑 `_features_dc` | ⭐⭐ |
| 调整大小 | 修改 `_scaling` | ⭐ |
| 替换外观 | PLY编辑后重载 | ⭐⭐⭐ |
| 添加新对象 | 创建GaussianModelActor并注册 | ⭐⭐⭐⭐ |

---

## 🔗 相关代码位置

- **模型定义**: `lib/models/street_gaussian_model.py`
- **对象类**: `lib/models/gaussian_model_actor.py`
- **位姿管理**: `lib/models/actor_pose.py`
- **PLY导出**: `lib/models/gaussian_model.py::save_ply()`
- **状态保存**: `lib/models/street_gaussian_model.py::save_state_dict()`

---

**版本**: v1.0 (精简版) | **更新**: 2026-08-13

> 💡 **提示**: 详细实施方案请参考完整文档或查看代码注释