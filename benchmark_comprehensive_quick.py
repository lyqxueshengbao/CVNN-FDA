"""
FDA-MIMO 综合评测脚本 (快速版本)
对比传统算法 (MUSIC, ESPRIT, OMP) 与深度学习方法 (CVNN)
用于快速测试和验证
"""
import os
import sys

# 修改参数以加快测试速度
QUICK_TEST_PARAMS = {
    'SNR_range': [-10, 0, 10, 20],  # 只测试4个SNR点
    'Monte_Carlo': 20,               # 减少蒙特卡洛次数
    'num_samples_cvnn': 200,         # 减少CVNN测试样本
    'Grid_theta_step': 2,            # 增大角度网格步长
    'Grid_r_step': 200,              # 增大距离网格步长
}

# 导入主评测脚本
import benchmark_comprehensive as bc
import numpy as np
import config as cfg

def main():
    print("\n" + "=" * 60)
    print("FDA-MIMO 快速评测模式")
    print("=" * 60)
    print("注意: 这是快速测试版本，结果可能不如完整版本准确")
    print("=" * 60)

    # 物理参数
    M = cfg.M
    N = cfg.N
    f0 = cfg.f0
    Delta_f = cfg.delta_f
    c0 = cfg.c
    lambda_ = cfg.wavelength
    d = cfg.d
    K = 1
    theta_true = 10.0 * np.pi / 180
    r_true = 2000.0
    L = 1

    # 快速测试的 SNR 范围
    SNR_dB_list = np.array(QUICK_TEST_PARAMS['SNR_range'])

    # 减少的蒙特卡洛次数
    Monte_Carlo_classical = QUICK_TEST_PARAMS['Monte_Carlo']
    num_samples_cvnn = QUICK_TEST_PARAMS['num_samples_cvnn']

    # 更粗的网格 (减少计算量)
    Grid_theta = np.arange(-50, 51, QUICK_TEST_PARAMS['Grid_theta_step'])
    Grid_r = np.arange(0, 5001, QUICK_TEST_PARAMS['Grid_r_step'])

    print(f"\n测试参数:")
    print(f"  SNR 点数: {len(SNR_dB_list)}")
    print(f"  蒙特卡洛次数 (传统算法): {Monte_Carlo_classical}")
    print(f"  测试样本数 (CVNN): {num_samples_cvnn}")
    print(f"  角度网格大小: {len(Grid_theta)}")
    print(f"  距离网格大小: {len(Grid_r)}")

    model_path = cfg.model_save_path
    save_path = 'results/comprehensive_benchmark_quick'
    os.makedirs(save_path, exist_ok=True)

    # ========== 计算 CRLB ==========
    print("\n" + "=" * 60)
    print("计算 CRLB...")
    print("=" * 60)

    crlb_theta_list = []
    crlb_r_list = []

    for SNR in SNR_dB_list:
        crlb_theta, crlb_r = bc.crlb_fda_mimo(
            theta_true, r_true, M, N, f0, Delta_f, c0, d, lambda_, L, SNR
        )
        crlb_theta_list.append(crlb_theta[0])
        crlb_r_list.append(crlb_r[0])

    crlb_results = {
        'snr_db_list': SNR_dB_list.tolist(),
        'crlb_theta': crlb_theta_list,
        'crlb_r': crlb_r_list
    }

    # ========== 评测传统算法 ==========
    print("\n" + "=" * 60)
    print("评测传统算法...")
    print("=" * 60)

    params = {
        'M': M, 'N': N, 'Delta_f': Delta_f, 'c0': c0, 'd': d, 'lambda_': lambda_,
        'K': K, 'L': L, 'theta_true': theta_true, 'r_true': r_true,
        'Grid_theta': Grid_theta, 'Grid_r': Grid_r
    }

    classical_results = []

    # 2D-MUSIC
    music_results = bc.evaluate_classical_algorithm(
        '2D-MUSIC', bc.music_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(music_results)

    # 2D-ESPRIT
    esprit_results = bc.evaluate_classical_algorithm(
        '2D-ESPRIT', bc.esprit_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(esprit_results)

    # OMP
    omp_results = bc.evaluate_classical_algorithm(
        'OMP', bc.omp_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(omp_results)

    # ========== 评测 CVNN ==========
    print("\n" + "=" * 60)
    print("评测 CVNN 模型...")
    print("=" * 60)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvnn_results = bc.evaluate_cvnn(
        model_path, SNR_dB_list, num_samples_cvnn,
        batch_size=64, device=device, model_type='standard'
    )

    # ========== 合并结果 ==========
    all_results = classical_results + [cvnn_results]

    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)

    import json
    results_dict = {
        'crlb': crlb_results,
        'algorithms': all_results,
        'parameters': {
            'M': M, 'N': N, 'f0': f0, 'Delta_f': Delta_f,
            'theta_true_deg': theta_true * 180 / np.pi,
            'r_true_m': r_true,
            'L': L,
            'Monte_Carlo_classical': Monte_Carlo_classical,
            'num_samples_cvnn': num_samples_cvnn,
            'quick_test': True
        }
    }

    with open(f'{save_path}/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"已保存结果: {save_path}/benchmark_results.json")

    # ========== 绘制对比图 ==========
    print("\n" + "=" * 60)
    print("绘制对比图...")
    print("=" * 60)

    bc.plot_comparison(all_results, crlb_results, save_path)

    # ========== 打印性能总结 ==========
    print("\n" + "=" * 60)
    print("性能总结")
    print("=" * 60)
    print(f"{'SNR (dB)':<10} {'Algorithm':<15} {'Angle RMSE':<15} {'Range RMSE':<15}")
    print("-" * 65)

    for snr_idx, snr in enumerate(SNR_dB_list):
        print(f"\n{snr:<10}")
        for result in all_results:
            algo = result['algorithm']
            angle_rmse = result['rmse_theta'][snr_idx]
            range_rmse = result['rmse_r'][snr_idx]
            print(f"{'':10} {algo:<15} {angle_rmse:<15.4f} {range_rmse:<15.4f}")

        # CRLB
        crlb_angle = crlb_results['crlb_theta'][snr_idx]
        crlb_range = crlb_results['crlb_r'][snr_idx]
        print(f"{'':10} {'CRLB':<15} {crlb_angle:<15.4f} {crlb_range:<15.4f}")

    print("\n" + "=" * 60)
    print("快速评测完成！")
    print("如需更准确的结果，请运行: python benchmark_comprehensive.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
