# 动态对象查询与导出工具 (obj_utils)

## 📁 文件结构

```
obj_utils/
├── query_obj.py          # 查询对象信息（182行）
├── export_obj.py         # 导出对象数据（270行）
├── EXPORT_ANALYSIS.md    # 导出对象完备性分析 ⭐ 新增
├── test_scripts.py       # 测试脚本
└── README.md             # 本文档
```

---

## 🎯 简介

`obj_utils/` 提供了两个核心工具脚本，用于查询和导出Waymo场景中的动态对象信息。

### 快速导航
- 🚀 **5分钟上手**: 见下方"快速开始"章节
- 📖 **完整说明**: 见下方"详细用法"章节
- 🔍 **导出完备性分析**: 见 [EXPORT_ANALYSIS.md](EXPORT_ANALYSIS.md) ⭐

---

## 🚀 快速开始

### Step 1: 探索场景

```bash
# 查看场景中有哪些动态对象
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml
```

**输出示例**:
```
📊 动态对象摘要 (7个对象)

📊 类别统计:
  - PEDESTRIAN: 3个 (IDs: [91, 109, 131])
  - VEHICLE: 4个 (IDs: [11, 17, 40, 84])

ID   类别       尺寸(m)            生命周期           
─────────────────────────────────────────────
11   VEHICLE    7.19×3.23×1.81    0-11 (12帧)     
17   VEHICLE    7.25×3.20×1.66    0-31 (32帧)     
```

### Step 2: 查看详情

```bash
# 查看特定对象的详细信息
python obj_utils/query_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11 \
    --verbose
```

### Step 3: 导出数据

```bash
# 导出单个对象（仅元数据，快速）
python obj_utils/export_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11 \
    --metadata-only

# 导出单个对象（完整数据，需要模型）
python obj_utils/export_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11
```

**生成结构**:
```
exports/031/
├── metadata/obj_011.json          # 静态元数据
├── trajectories/traj_011.json     # 动态轨迹
├── pth/obj_011.pth                # 高斯参数+位姿
└── ply/
    ├── obj_011.ply                # PLY点云
    └── obj_011_meta.json          # PLY元数据
```

### Step 4: 编辑对象（可选）

```python
import torch

# 加载导出的对象
obj = torch.load('exports/031/pth/obj_011.pth')

# 修改颜色（让车变红）
features_dc = obj['gaussian_params']['feature_dc']
features_dc[:, 0, :] *= 1.5  # 增强红色通道

# 保存修改
torch.save(obj, 'obj_011_red.pth')
```

---

## 🔍 详细用法

### 查询工具 (query_obj.py)

#### 基本用法

```bash
# 查看所有对象摘要
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml

# 查看指定对象详情
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --verbose

# 仅查看轨迹
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml --mode trajectory

# 仅查看元数据
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml --mode metadata
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `--config` | str | **必需**，配置文件路径 |
| `--track_id` | int[] | 指定要查询的对象ID列表（不指定则显示所有） |
| `--verbose` | flag | 详细模式（显示完整信息） |
| `--mode` | str | 查询模式：`metadata` / `trajectory` / `both`（默认） |

#### 输出内容

**摘要模式**:
- 📊 类别统计汇总
- 每个对象的基本信息（ID、类别、尺寸、生命周期）

**详细模式**:
- 完整的对象元数据（类别、标签、可变形性）
- 精确的尺寸信息（长宽高、体积）
- 时间范围（起始帧、结束帧、持续时间）

**轨迹模式**:
- 出现帧数和帧范围
- 移动距离和平均速度
- 轨迹采样点（位置、偏航角）

---

### 导出工具 (export_obj.py)

#### 基本用法

```bash
# 导出所有对象
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --all

# 导出指定对象
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 17 40

# 仅导出元数据（快速，不需要模型）
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --metadata-only

# 指定输出目录
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --output_dir ./my_exports

# PTH不包含位姿（减小文件大小）
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --no-pose

# 跳过PLY导出（仅导出PTH）
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --skip-ply
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `--config` | str | **必需**，配置文件路径 |
| `--track_id` | int[] | 指定要导出的对象ID列表 |
| `--all` | flag | 导出所有对象 |
| `--output_dir` | str | 输出根目录（默认：`./exports`） |
| `--metadata-only` | flag | 仅导出元数据和轨迹（速度快） |
| `--no-pose` | flag | PTH导出时不包含位姿轨迹 |
| `--skip-ply` | flag | 跳过PLY导出 |

#### 输出目录结构

```
exports/{waymo_sequence_id}/
├── metadata/                  # 对象静态元数据
│   └── obj_{track_id:03d}.json
├── trajectories/              # 对象动态轨迹
│   └── traj_{track_id:03d}.json
├── pth/                       # 高斯参数（含位姿+Fourier）
│   └── obj_{track_id:03d}.pth
└── ply/                       # PLY点云格式
    ├── obj_{track_id:03d}.ply
    └── obj_{track_id:03d}_meta.json
```

#### 文件格式详解

**JSON元数据** (`metadata/obj_XXX.json`):
```json
{
  "track_id": 11,
  "class": "vehicle",
  "class_label": 0,
  "height": 1.81,
  "width": 3.23,
  "length": 7.19,
  "deformable": false,
  "start_frame": 0,
  "end_frame": 11,
  "start_timestamp": 0.0,
  "end_timestamp": 1.2
}
```

**JSON轨迹** (`trajectories/traj_XXX.json`):
```json
{
  "track_id": 11,
  "class": "vehicle",
  "dimensions": {
    "length": 7.19,
    "width": 3.23,
    "height": 1.81
  },
  "time_range": {
    "start_frame": 0,
    "end_frame": 11,
    "duration_frames": 12
  },
  "trajectory": [
    {
      "frame": 0,
      "position": {"x": 17.19, "y": 2.78, "z": 0.72},
      "orientation": {
        "quaternion": [0.009, 0.0, 0.0, 1.0],
        "yaw_rad": 3.124,
        "yaw_deg": 179.01
      }
    }
  ]
}
```

**PTH高斯参数** (`pth/obj_XXX.pth`):
```python
{
    'gaussian_params': {
        'xyz': Tensor[N, 3],           # 局部坐标位置
        'feature_dc': Tensor[N, 3, D], # Fourier SH DC分量
        'feature_rest': Tensor[N, 3, K], # Fourier SH高阶分量
        'scaling': Tensor[N, 3],       # 缩放
        'rotation': Tensor[N, 4],      # 旋转（四元数）
        'opacity': Tensor[N, 1],       # 不透明度
        'semantic': Tensor[N, C],      # 语义标签
    },
    'obj_meta': {...},                 # 对象元数据
    'fourier_config': {                # Fourier SH配置
        'fourier_dim': 5,
        'fourier_scale': 1.0,
    },
    'pose_trajectory': {               # 位姿轨迹（可选）
        'opt_trans': [...],
        'opt_rots': [...],
        'track_idx': [...],
        'timestamps': [...],
    }
}
```

**PLY点云** (`ply/obj_XXX.ply`):
- 标准PLY格式，可用以下软件打开：
  - **Blender**（需安装Point Cloud插件）
  - **CloudCompare**（免费开源）
  - **MeshLab**（轻量级）
  - **Open3D**（Python库）

---

## 💡 典型工作流

### 场景1: 探索并导出感兴趣的对象

```bash
# Step 1: 查看场景中有哪些对象
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml

# Step 2: 查看某个对象的详细信息
python obj_utils/query_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11 \
    --verbose

# Step 3: 导出该对象
python obj_utils/export_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11

# Step 4: 在Blender中编辑PLY文件
blender exports/031/ply/obj_011.ply
```

### 场景2: 批量处理多个对象

```bash
# 一次性导出多个对象
python obj_utils/export_obj.py \
    --config configs/example/waymo_train_031.yaml \
    --track_id 11 17 40 84 \
    --skip-ply  # 跳过慢速的PLY导出
```

### 场景3: 程序化编辑对象

```python
import torch

# 加载导出的对象
obj = torch.load('exports/031/pth/obj_011.pth')

# 修改颜色（让车变红）
features_dc = obj['gaussian_params']['feature_dc']
features_dc[:, 0, :] *= 1.5  # 增强红色通道
features_dc[:, 1:, :] *= 0.5  # 减弱绿蓝通道

# 保存修改
torch.save(obj, 'obj_011_red.pth')
```

### 场景4: 重新集成编辑后的对象

参考 [`docs/OBJECT_DECOUPLING_ANALYSIS.md`](docs/OBJECT_DECOUPLING_ANALYSIS.md) 中的"替换对象"章节。

---

## ❓ 常见问题

### Q1: 如何知道应该导出哪个对象？
**A:** 先用 `query_obj.py` 查看所有对象，找到感兴趣的ID后再导出。

### Q2: 导出很慢怎么办？
**A:** 
- 使用 `--metadata-only` 跳过模型加载（秒级完成）
- 使用 `--skip-ply` 跳过PLY导出（PLY较慢）
- 使用 `--no-pose` 减小PTH文件大小

### Q3: PLY文件在哪里打开？
**A:** 
- **Blender**: 安装Point Cloud Viewer插件
- **CloudCompare**: 免费开源的点云查看器
- **MeshLab**: 轻量级3D网格编辑器
- **Open3D**: Python库，可编程操作

### Q4: 如何重新集成编辑后的对象？
**A:** 参考 [`docs/OBJECT_DECOUPLING_ANALYSIS.md`](docs/OBJECT_DECOUPLING_ANALYSIS.md) 中的"替换对象"章节，使用 `replace_object_in_model()` 函数。

### Q5: 内存不足（OOM）怎么办？
**A:** 
- `query_obj.py` 已优化，不会OOM（不加载图像）
- 如果 `export_obj.py` OOM，尝试：
  - 减少并发或关闭其他GPU应用
  - 使用 `--metadata-only` 跳过模型加载
  - 分批导出对象

### Q6: PTH和PLY有什么区别？
**A:** 
- **PTH**: Python pickle格式，包含完整参数（高斯属性+位姿+Fourier配置），适合程序化编辑
- **PLY**: 标准点云格式，仅包含几何信息，适合3D软件可视化编辑

### Q7: 导出的文件很大怎么办？
**A:** 
- 使用 `--no-pose` 排除位姿轨迹
- 使用 `--skip-ply` 跳过PLY导出
- 考虑压缩或使用更低的SH阶数

---

## 🎓 学习路径

### 初学者
1. ✅ 阅读上方"快速开始"章节
2. ✅ 运行测试脚本: `python obj_utils/test_scripts.py --config configs/example/waymo_train_031.yaml`
3. ✅ 尝试导出一个对象并在Blender中查看

### 进阶用户
1. 📖 阅读上方"详细用法"了解所有参数
2. 🔧 学习程序化编辑方法（修改PTH文件）
3. ⚡ 探索批量处理技巧

### 高级用户
1. 📚 阅读 [`docs/OBJECT_DECOUPLING_ANALYSIS.md`](docs/OBJECT_DECOUPLING_ANALYSIS.md)
2. 💻 实现自定义的对象操作（添加、删除、合并）
3. 🤝 贡献新的工具脚本

---

## 📊 性能优化建议

| 场景 | 推荐命令 | 原因 |
|------|---------|------|
| 快速浏览对象 | `query_obj.py --mode metadata` | 不加载轨迹数据，速度更快 |
| 仅需元数据 | `export_obj.py --metadata-only` | 跳过高斯模型导出，节省大量时间 |
| 程序化编辑 | 导出PTH格式 | 保留完整参数和位姿信息 |
| 可视化编辑 | 导出PLY格式 | 兼容Blender等3D软件 |
| 大场景处理 | `export_obj.py --skip-ply` | PLY导出较慢，可仅用PTH |

---

## ⚠️ 注意事项

1. **内存管理**: `query_obj.py` 使用 `sceneLoadTypeCallbacks` 直接加载元数据，避免Dataset类自动加载所有图像导致OOM

2. **模型依赖**: 导出PTH/PLY需要训练好的模型（`.pth`文件），确保 `cfg.model_path` 配置正确

3. **Fourier SH时间维度**: 动态对象的外观随时间变化，编辑时需注意 `fourier_dim` 一致性

4. **坐标系**: PTH中的高斯点使用局部坐标系，渲染时需通过ActorPose转换为世界坐标

5. **备份原模型**: 编辑对象并重新集成前，务必备份原始模型文件

6. **验证完整性**: 替换对象后应测试渲染效果，必要时进行少量迭代微调

---

## ✅ 测试验证

```bash
# 测试查询
python obj_utils/query_obj.py --config configs/example/waymo_train_031.yaml
# ✅ 输出: 7个对象（4车+3行人）

# 测试导出
python obj_utils/export_obj.py --config configs/example/waymo_train_031.yaml --track_id 11 --metadata-only
# ✅ 输出: exports/031/metadata/obj_011.json + trajectories/traj_011.json
```

---

## 🔗 相关资源

- **对象解耦详细方案**: [`docs/OBJECT_DECOUPLING_ANALYSIS.md`](docs/OBJECT_DECOUPLING_ANALYSIS.md)
- **快速参考卡片**: [`docs/OBJECT_DECOUPLING_QUICKREF.md`](docs/OBJECT_DECOUPLING_QUICKREF.md)
- **项目主页**: [`README.md`](README.md)
- **训练指南**: [`script/train_waymo_example.sh`](script/train_waymo_example.sh)
- **渲染指南**: [`script/render_waymo_example.sh`](script/render_waymo_example.sh)

---

## 📝 版本历史

### v2.0 (2026-08-14) - 当前版本
- ✨ 精简代码：query_obj.py (433→182行), export_obj.py (561→270行)
- ✨ 整合文档：三个文档合并为一个README.md
- ✨ 完善测试：使用 `configs/example/waymo_train_031.yaml` 验证通过
- ⚡ 内存优化：查询时不加载图像，避免OOM

### v2.0 (2026-08-13)
- ✨ 整合 `metadata_info.py` 和 `model_info.py` 为两个专业脚本
- ✨ 新增标准化导出目录结构（按Waymo ID分类）
- ✨ 完善命令行参数和错误处理
- ✨ 添加详细文档和快速开始指南
- ⚡ 内存优化：查询时不加载图像

### v1.0 (之前)
- 原始的 `metadata_info.py` 和 `model_info.py` 脚本

---

**最后更新**: 2026-08-14  
**维护者**: StreetGaussians Team  
**测试配置**: `configs/example/waymo_train_031.yaml`  
**测试结果**: ✅ 全部通过
