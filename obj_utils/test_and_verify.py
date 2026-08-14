#!/usr/bin/env python3
"""
obj_utils 工具测试与验证套件

功能：
1. 测试 query_obj.py 和 export_obj.py 的基本功能
2. 验证导出对象的完备性
3. 提供快速回归测试

用法：
    # 完整测试（查询+导出+验证）
    python obj_utils/test_and_verify.py --config configs/example/waymo_train_031.yaml
    
    # 仅测试查询功能
    python obj_utils/test_and_verify.py --config configs/example/waymo_train_031.yaml --skip-export
    
    # 仅验证已有导出文件
    python obj_utils/test_and_verify.py --verify exports/031/pth/obj_011.pth
"""

import os
import sys
import subprocess
import argparse
import torch

# 确保项目根目录在sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_command(cmd, description, timeout=300):
    """运行命令并返回结果"""
    print(f"\n{'=' * 80}")
    print(f"🧪 测试: {description}")
    print(f"{'=' * 80}")
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("✅ 测试通过")
            # 显示关键输出（前30行）
            lines = result.stdout.split('\n')
            for line in lines[:30]:
                if line.strip():
                    print(line)
            if len(lines) > 30:
                print(f"... (还有 {len(lines) - 30} 行)")
            return True, result.stdout
        else:
            print("❌ 测试失败")
            print(f"错误输出:\n{result.stderr[-500:]}")  # 只显示最后500字符
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("⏱️  测试超时")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False, str(e)


def verify_export(obj_path):
    """验证导出对象的完备性"""
    print(f"\n{'=' * 80}")
    print(f"🔍 验证导出对象: {obj_path}")
    print(f"{'=' * 80}\n")
    
    if not os.path.exists(obj_path):
        print(f"❌ 文件不存在: {obj_path}")
        return False
    
    try:
        obj_state = torch.load(obj_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 检查清单
    checks = {
        'gaussian_params': 'gaussian_params' in obj_state,
        'obj_meta': 'obj_meta' in obj_state,
        'fourier_config': 'fourier_config' in obj_state,
        'local_frame': 'local_frame' in obj_state,
        'pose_trajectory': 'pose_trajectory' in obj_state,
        'bounding_box': 'bounding_box' in obj_state,
        'statistics': 'statistics' in obj_state,
        'version': 'version' in obj_state,
    }
    
    # P0必需项
    p0_checks = ['local_frame', 'pose_trajectory', 'bounding_box']
    p0_pass = sum(1 for k in p0_checks if checks[k])
    p0_total = len(p0_checks)
    
    # 打印检查结果
    print("📋 完备性检查:\n")
    for key, passed in checks.items():
        status = "✅" if passed else "❌"
        priority = ""
        if key in p0_checks:
            priority = " (P0必需)"
        elif key == 'statistics':
            priority = " (P1推荐)"
        elif key == 'version':
            priority = " (P2可选)"
        print(f"{status} {key}{priority}")
    
    # 详细信息
    if checks.get('gaussian_params'):
        params = obj_state['gaussian_params']
        n_pts = params['xyz'].shape[0] if isinstance(params['xyz'], torch.Tensor) else len(params['xyz'])
        print(f"\n   高斯点数: {n_pts}")
    
    if checks.get('pose_trajectory'):
        pose = obj_state['pose_trajectory']
        has_full = 'input_trans' in pose and 'input_rots' in pose
        n_frames = len(pose.get('frame_indices', []))
        print(f"   位姿帧数: {n_frames}, 完整位姿: {'✓' if has_full else '✗'}")
    
    if checks.get('bounding_box'):
        bbox = obj_state['bounding_box']
        print(f"   边界框: {bbox.get('type', '?')}, 体积: {bbox.get('volume_m3', '?')} m³")
    
    # 评分
    print(f"\n{'=' * 80}")
    print(f"P0必需项: {p0_pass}/{p0_total} {'✅' if p0_pass == p0_total else '❌'}")
    
    score = (p0_pass / p0_total) * 5
    print(f"完备性评分: {score:.1f}/5.0 {'⭐' * int(score)}")
    print(f"{'=' * 80}")
    
    if p0_pass == p0_total:
        print("\n✅ 导出对象完备！可用于场景编辑和重组")
        return True
    else:
        missing = [k for k in p0_checks if not checks[k]]
        print(f"\n❌ 缺少P0必需项: {missing}")
        return False


def main():
    parser = argparse.ArgumentParser(description='obj_utils 测试与验证套件')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--track_id', type=int, default=11, help='测试用的对象ID')
    parser.add_argument('--output_dir', type=str, default='./test_exports', help='测试输出目录')
    parser.add_argument('--skip-export', action='store_true', help='跳过导出测试')
    parser.add_argument('--verify', type=str, default=None, help='验证指定的导出文件')
    
    args = parser.parse_args()
    
    # 如果只验证文件
    if args.verify:
        success = verify_export(args.verify)
        sys.exit(0 if success else 1)
    
    # 否则执行完整测试流程
    if not args.config:
        print("❌ 错误: 必须指定 --config 或 --verify")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 obj_utils 测试与验证套件")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"测试对象ID: {args.track_id}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    results = []
    
    # 测试1: query_obj.py - 概括模式
    cmd1 = [
        sys.executable, 'obj_utils/query_obj.py',
        '--config', args.config
    ]
    passed, _ = run_command(cmd1, "query_obj.py - 概括模式")
    results.append(("query_summary", passed))
    
    # 测试2: query_obj.py - 详细模式
    cmd2 = [
        sys.executable, 'obj_utils/query_obj.py',
        '--config', args.config,
        '--track_id', str(args.track_id),
        '--verbose'
    ]
    passed, _ = run_command(cmd2, f"query_obj.py - 详细模式 (ID={args.track_id})")
    results.append(("query_detail", passed))
    
    if not args.skip_export:
        # 测试3: export_obj.py - 元数据导出
        cmd3 = [
            sys.executable, 'obj_utils/export_obj.py',
            '--config', args.config,
            '--track_id', str(args.track_id),
            '--metadata-only'
        ]
        passed, _ = run_command(cmd3, "export_obj.py - 仅元数据")
        results.append(("export_metadata", passed))
        
        # 测试4: export_obj.py - 完整导出
        cmd4 = [
            sys.executable, 'obj_utils/export_obj.py',
            '--config', args.config,
            '--track_id', str(args.track_id),
            '--output_dir', args.output_dir
        ]
        passed, output = run_command(cmd4, f"export_obj.py - 完整导出 (ID={args.track_id})", timeout=600)
        results.append(("export_full", passed))
        
        # 测试5: 验证导出的完备性
        if passed:
            obj_path = os.path.join(args.output_dir, os.path.basename(args.config).replace('.yaml', ''), 
                                   'pth', f'obj_{args.track_id:03d}.pth')
            if os.path.exists(obj_path):
                verified = verify_export(obj_path)
                results.append(("verify_completeness", verified))
            else:
                print(f"\n⚠️ 未找到导出文件: {obj_path}")
                results.append(("verify_completeness", False))
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    failed_count = total - passed_count
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n总计: {total} 个测试")
    print(f"✅ 通过: {passed_count}")
    print(f"❌ 失败: {failed_count}")
    
    if failed_count == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ 有 {failed_count} 个测试失败，请检查上方输出")
    
    print("=" * 80)
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
