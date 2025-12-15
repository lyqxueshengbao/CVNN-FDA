"""
批量运行所有算法的评测
"""
import subprocess
import argparse
import time

def run_all_benchmarks(L_snapshots=10, num_samples=500, snr_list=None, parallel=False):
    """
    运行所有算法的评测

    Args:
        L_snapshots: 快拍数
        num_samples: 样本数
        snr_list: SNR列表
        parallel: 是否并行运行
    """
    if snr_list is None:
        snr_list = [-5, 0, 5, 10, 15, 20]

    snr_str = ' '.join([str(s) for s in snr_list])

    # 定义所有评测任务
    tasks = [
        {
            'name': 'CVNN',
            'cmd': f'python benchmark_cvnn.py --L {L_snapshots} --samples {num_samples} --snr-list {snr_str}'
        },
        {
            'name': 'MUSIC (standard)',
            'cmd': f'python benchmark_music.py --L {L_snapshots} --samples {num_samples} --grid standard --snr-list {snr_str}'
        },
        {
            'name': 'Capon (standard)',
            'cmd': f'python benchmark_capon.py --L {L_snapshots} --samples {num_samples} --grid standard --snr-list {snr_str}'
        },
    ]

    print("="*70)
    print(f"批量运行评测")
    print("="*70)
    print(f"快拍数: L={L_snapshots}")
    print(f"样本数: {num_samples}")
    print(f"SNR范围: {snr_list}")
    print(f"并行模式: {'是' if parallel else '否'}")
    print(f"\n共 {len(tasks)} 个任务:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task['name']}")
    print("="*70)

    if parallel:
        # 并行运行
        print("\n🚀 并行启动所有任务...")
        processes = []
        for task in tasks:
            print(f"  启动: {task['name']}")
            p = subprocess.Popen(task['cmd'], shell=True)
            processes.append((task['name'], p))

        # 等待所有任务完成
        print("\n⏳ 等待所有任务完成...")
        for name, p in processes:
            p.wait()
            if p.returncode == 0:
                print(f"  ✅ {name} 完成")
            else:
                print(f"  ❌ {name} 失败 (返回码: {p.returncode})")

    else:
        # 串行运行
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*70}")
            print(f"运行任务 {i}/{len(tasks)}: {task['name']}")
            print(f"{'='*70}")

            start_time = time.time()
            result = subprocess.run(task['cmd'], shell=True)
            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"\n✅ {task['name']} 完成 (耗时: {elapsed:.1f}秒)")
            else:
                print(f"\n❌ {task['name']} 失败 (返回码: {result.returncode})")

    # 合并结果
    print(f"\n{'='*70}")
    print("合并结果...")
    print(f"{'='*70}")
    merge_cmd = f'python merge_results.py --L {L_snapshots}'
    subprocess.run(merge_cmd, shell=True)

    print(f"\n{'='*70}")
    print("✅ 所有评测完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量运行所有算法评测')
    parser.add_argument('--L', type=int, default=10,
                        help='快拍数')
    parser.add_argument('--samples', type=int, default=500,
                        help='每个SNR的样本数')
    parser.add_argument('--snr-list', type=float, nargs='+',
                        default=[-5, 0, 5, 10, 15, 20],
                        help='SNR列表')
    parser.add_argument('--parallel', action='store_true',
                        help='并行运行所有任务')

    args = parser.parse_args()

    run_all_benchmarks(
        L_snapshots=args.L,
        num_samples=args.samples,
        snr_list=args.snr_list,
        parallel=args.parallel
    )
