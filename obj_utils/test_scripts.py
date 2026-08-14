#!/usr/bin/env python3
"""
obj_utils工具脚本快速测试

用法：
    python obj_utils/test_scripts.py --config configs/experiments_waymo/waymo_val_006.yaml
"""

import os
import sys
import subprocess
import argparse

# 确保项目根目录在sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'=' * 80}")
    print(f"🧪 测试: {description}")
    print(f"{'=' * 80}")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 测试通过")
        if result.stdout:
            # 只显示前50行输出
            lines = result.stdout.split('\n')
            for line in lines[:50]:
                print(line)
            if len(lines) > 50:
                print(f"... (还有 {len(lines) - 50} 行)")
    else:
        print("❌ 测试失败")
        print(f"错误输出:\n{result.stderr}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='测试obj_utils脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--track_id', type=int, default=5, help='测试用的对象ID')
    parser.add_argument('--skip-export', action='store_true', help='跳过导出测试（仅测试查询）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 obj_utils 工具脚本测试套件")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"测试对象ID: {args.track_id}")
    print()
    
    results = []
    
    # 测试1: query_obj.py - 概括模式
    cmd1 = [
        sys.executable, 'obj_utils/query_obj.py',
        '--config', args.config
    ]
    results.append(run_command(cmd1, "query_obj.py - 概括模式"))
    
    # 测试2: query_obj.py - 详细模式（指定对象）
    cmd2 = [
        sys.executable, 'obj_utils/query_obj.py',
        '--config', args.config,
        '--track_id', str(args.track_id),
        '--verbose'
    ]
    results.append(run_command(cmd2, f"query_obj.py - 详细模式 (对象 {args.track_id})"))
    
    # 测试3: query_obj.py - 仅轨迹模式
    cmd3 = [
        sys.executable, 'obj_utils/query_obj.py',
        '--config', args.config,
        '--mode', 'trajectory'
    ]
    results.append(run_command(cmd3, "query_obj.py - 仅轨迹模式"))
    
    if not args.skip_export:
        # 测试4: export_obj.py - 仅元数据
        cmd4 = [
            sys.executable, 'obj_utils/export_obj.py',
            '--config', args.config,
            '--track_id', str(args.track_id),
            '--metadata-only'
        ]
        results.append(run_command(cmd4, f"export_obj.py - 仅元数据 (对象 {args.track_id})"))
        
        # 测试5: export_obj.py - 完整导出
        cmd5 = [
            sys.executable, 'obj_utils/export_obj.py',
            '--config', args.config,
            '--track_id', str(args.track_id),
            '--output_dir', './test_exports'
        ]
        results.append(run_command(cmd5, f"export_obj.py - 完整导出 (对象 {args.track_id})"))
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"总测试数: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查上方输出")
    
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
