# 常见错误与解决方案

## 🐛 错误记录

### 错误 1: TypeError in get_val_frames - split_test/split_train 配置错误

**发生时间**: 2024-08-17  
**场景**: 使用 `setup_render_model.py` 生成的配置文件运行渲染时

#### ❌ 错误现象

```bash
python render.py --config models/031/render_031.yaml
```

**错误信息**:
```
TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
  File "/mnt/workspace/street_gaussians/lib/utils/data_utils.py", line 38
    val_frames = set(np.arange(test_every, num_frames, test_every))
```

#### 🔍 根本原因

在 [`lib/utils/data_utils.py`](../lib/utils/data_utils.py#L38) 的 `get_val_frames` 函数中：

```python
def get_val_frames(num_frames: int, test_every: int, train_every: int):
    if train_every is None or train_every < 0:
        # 当 test_every 为 None 时，执行 np.arange(None, num_frames, None) 会报错
        val_frames = set(np.arange(test_every, num_frames, test_every))
```

**问题**: 代码期望的参数值是：
- `-1`: 表示不分割（所有帧都用于某一目的）
- **正整数**: 每隔 N 帧采样
- **不能是 `None`**: 会导致 `np.arange()` 报错

但最初生成的配置文件使用了 `null`（在 YAML 中被解析为 Python 的 `None`），导致错误。

#### ✅ 正确配置

**参考可运行的配置文件** ([`configs/experiments_waymo/waymo_val_006.yaml`](../configs/experiments_waymo/waymo_val_006.yaml)):

```yaml
data:
  split_test: -1      # ✅ 正确：-1 表示不分割测试集
  split_train: -1     # ✅ 正确：-1 表示不分割训练集
```

**其他有效配置示例**:

```yaml
# 示例 1: 渲染所有帧（无分割）
data:
  split_test: -1
  split_train: -1

# 示例 2: 每 4 帧渲染一次（测试集）
data:
  split_test: 4
  split_train: -1

# 示例 3: 训练集每帧都用，测试集每 5 帧
data:
  split_test: 5
  split_train: 1
```

#### 📝 修复步骤

1. **编辑配置文件**:
```bash
vim models/031/render_031.yaml
```

2. **确保使用 -1 而不是 null**:
```yaml
data:
  split_test: -1    # ✅ 正确
  split_train: -1   # ✅ 正确
  # ❌ 不要使用: split_test: null
```

3. **重新运行**:
```bash
python render.py --config models/031/render_031.yaml mode evaluate
```

#### 🔧 已修复

[`script/setup_render_model.py`](../script/setup_render_model.py) 已更新，现在生成的配置文件默认使用 `-1`：

```python
# 修复后的代码
data:
  split_test: -1    # ✅ 正确
  split_train: -1   # ✅ 正确
```

#### ⚠️ 注意事项

1. **参数含义**:
   - `-1`: 不分割，所有帧都包含
   - `1`: 每 1 帧采样（即所有帧）
   - `N > 1`: 每 N 帧采样一次

2. **训练 vs 渲染配置**:
   ```yaml
   # 训练配置 (configs/example/waymo_train_031.yaml)
   data:
     split_test: -1      # 测试集不参与训练
     split_train: 1      # 训练集使用所有帧
   
   # 渲染配置 (configs/experiments_waymo/waymo_val_006.yaml)
   data:
     split_train: -1     # 不分割训练集
     split_test: 4       # 每 4 帧渲染一次
   ```

3. **验证配置**:
   ```python
   from lib.config import cfg
   cfg.merge_from_file('models/031/render_031.yaml')
   
   assert cfg.data.split_test == -1, "split_test 应该是 -1"
   assert cfg.data.split_train == -1, "split_train 应该是 -1"
   print("✅ 配置正确")
   ```

#### 📚 相关代码位置

- **参数处理**: [`lib/utils/data_utils.py:35-40`](../lib/utils/data_utils.py#L35-L40)
- **调用链**: 
  - [`render.py:20`](../render.py#L20) → 
  - [`lib/datasets/dataset.py:30`](../lib/datasets/dataset.py#L30) → 
  - [`lib/utils/waymo_utils.py:330`](../lib/utils/waymo_utils.py#L330) → 
  - [`lib/utils/data_utils.py:35`](../lib/utils/data_utils.py#L35)

---

## 📖 更新日志

- **2024-08-17**: 
  - 添加 `split_test/split_train` 配置错误记录
  - 修复 `setup_render_model.py` 使用正确的 `-1` 值
  - 更新配置文件 `models/031/render_031.yaml`

---

*最后更新: 2024-08-17*
