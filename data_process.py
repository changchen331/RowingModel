import csv

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# 获取数据
raw_vb = []  # 船速（m/s）
with open("data/boat_speed.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        raw_vb.append((float(row[0]), float(row[1])))

raw_F = []  # 桨柄力（N）
with open("data/hand_force.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        raw_F.append((float(row[0]), float(row[1])))

raw_xBF = []  # 腿位移（m）
with open("data/leg_displacement.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        raw_xBF.append((float(row[0]), float(row[1])))

raw_xSB = []  # 背位移（m）
with open("data/back_displacement.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        raw_xSB.append((float(row[0]), float(row[1])))

raw_theta = []  # 桨角（°）
with open("data/oar_angle.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        raw_theta.append((float(row[0]), float(row[1])))


def unify_time_axis(raw_data, n_points=50):
    """
    将不等间隔的原始数据统一到n_points个等间隔点。

    Parameters
    ----------
    raw_data : list of (t, value) tuples
    n_points : int，目标点数（论文使用50）

    Returns
    -------
    t_uniform : 均匀时间轴
    v_uniform : 对应的插值数值
    cs        : CubicSpline对象（可继续求导）
    """
    # 拆分时间和数值
    t_raw = np.array([p[0] for p in raw_data])
    v_raw = np.array([p[1] for p in raw_data])

    # 确保时间轴单调递增（排序）
    sort_idx = np.argsort(t_raw)
    t_raw = t_raw[sort_idx]
    v_raw = v_raw[sort_idx]

    # 确定时间范围
    t_start = t_raw[0]
    t_end = t_raw[-1]
    T = t_end - t_start  # 划桨周期

    # 构建三次样条（用原始不均匀点）
    cs = CubicSpline(t_raw, v_raw, bc_type='periodic')

    # 在均匀时间轴上重新采样
    t_uniform = np.linspace(t_start, t_end, n_points)
    v_uniform = cs(t_uniform)

    return t_uniform, v_uniform, cs, T


def process_data():
    # ============================================================
    # 对5个变量分别处理
    # ============================================================
    n_points = 1000
    t_vb, vb, cs_vb, T_vb = unify_time_axis(raw_vb, n_points)
    t_F, F, cs_F, T_F = unify_time_axis(raw_F, n_points)
    t_xBF, xBF, cs_xBF, T_xBF = unify_time_axis(raw_xBF, n_points)
    t_xSB, xSB, cs_xSB, T_xSB = unify_time_axis(raw_xSB, n_points)
    t_theta, theta, cs_theta, T_theta = unify_time_axis(raw_theta, n_points)

    # ============================================================
    # 统一到同一个时间轴（取交集范围）
    # ============================================================

    # 各传感器的时间范围可能略有不同
    # 取所有数据共同覆盖的时间范围
    t_start_common = max(t_vb[0], t_F[0], t_xBF[0], t_xSB[0], t_theta[0])
    t_end_common = min(t_vb[-1], t_F[-1], t_xBF[-1], t_xSB[-1], t_theta[-1])

    # 最终统一时间轴
    t_common = np.linspace(t_start_common, t_end_common, n_points)

    # 用各自的样条在统一时间轴上求值
    vb_final = cs_vb(t_common)
    F_final = cs_F(t_common)
    xBF_final = cs_xBF(t_common)
    xSB_final = cs_xSB(t_common)
    theta_final = cs_theta(t_common)

    return vb_final, F_final, xBF_final, xSB_final, theta_final, t_common


if __name__ == '__main__':
    v_b, f, x_BF, x_SB, angle, t = process_data()
    datasets = [
        (raw_vb, v_b, '船速 (m/s)'),
        (raw_F, f, '桨柄力 (N)'),
        (raw_xBF, x_BF, '腿位移 (m)'),
        (raw_xSB, x_SB, '背位移 (m)'),
        (raw_theta, angle, '桨角 (°)'),
    ]

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for ax, (raw, unified, label) in zip(axes, datasets):
        t_r = [p[0] for p in raw]
        v_r = [p[1] for p in raw]

        # 原始数据（散点）
        ax.scatter(t_r, v_r, s=10, color='gray', alpha=0.5, label='原始数据')

        # 插值后均匀数据（线）
        ax.plot(t, unified, color='blue', linewidth=1.5, label='插值结果')

        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间 (s)')
    plt.tight_layout()
    # plt.savefig('time_axis_unification.png', dpi=150)
    plt.show()
