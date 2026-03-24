import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from data_process import process_data

# ============================================================
# 模型常数（Table 1，单人艇）
# ============================================================
PARAMS = {
    'T': 1.94,  # 划桨周期 (s)
    'mR': 75.0,  # 划桨者质量 (kg)
    'mb': 19.7,  # 船体质量 (kg)
    'mO': 1.2,  # 桨质量 (kg)
    's': 0.83,  # 内侧桨长 (m)
    'l': 1.805,  # 外侧桨长 (m)
    'C1': 3.16,  # 船体阻力系数
    'C2': 58.7,  # 桨叶阻力系数
    'r': 0.4,  # 质心高度比
    'd': 0.565,  # 桨锁到桨质心距离 (m)
    'IG': 0.85,  # 桨转动惯量 (kg·m²)
    'dLF': 0.0,  # 桨锁前后位置（初始估计）
}


# ============================================================
# 第一步：准备协调运动样条
# ============================================================
def prepare_splines(x_bf, x_sb, theta, t_common, params):
    """
    由实测数据构建协调运动样条，并反推x_H/S

    Returns
    -------
    cs_leg, cs_trunk, cs_arm : CubicSpline对象
    """
    s = params['s']
    dLF = params['dLF']

    # 转换为相对位移
    x_bf = x_bf - x_bf[0]
    x_sb = x_sb - x_sb[0]

    # 由Eq.8反推 x_H/S
    xHS = s * np.sin(theta) - dLF + x_sb + x_bf

    # 构建三次样条（不要求周期性，因为是实测数据）
    cs_leg = CubicSpline(t_common, x_bf)
    cs_trunk = CubicSpline(t_common, x_sb)
    cs_arm = CubicSpline(t_common, xHS)

    return cs_leg, cs_trunk, cs_arm


# ============================================================
# 第二步：由样条计算桨角及导数（Eq.8, 9）
# ============================================================
def compute_theta_from_splines(t, cs_leg, cs_trunk, cs_arm, params):
    """
    由协调运动样条计算桨角θ及其导数

    注意：这里用实测θ的样条直接求导更准确
    此函数作为备用验证
    """
    s = params['s']
    dLF = params['dLF']

    xBF = cs_leg(t)
    xSB = cs_trunk(t)
    xHS = cs_arm(t)

    xBFd = cs_leg.derivative(1)(t)
    xSBd = cs_trunk.derivative(1)(t)
    xHSd = cs_arm.derivative(1)(t)

    xBFdd = cs_leg.derivative(2)(t)
    xSBdd = cs_trunk.derivative(2)(t)
    xHSdd = cs_arm.derivative(2)(t)

    # Eq.8
    sin_theta = (dLF + xHS - xSB - xBF) / s
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)

    # Eq.9：θ̇
    theta_dot = (xHSd - xBFd - xSBd) / (s * cos_theta)

    # Eq.9：θ̈
    theta_ddot = ((xHSdd - xBFdd - xSBdd) + s * theta_dot ** 2 * np.sin(theta)) / (s * cos_theta)

    return theta, theta_dot, theta_ddot


# ============================================================
# 第三步：判断Drive/Recovery（Eq.16）
# ============================================================
def blade_normal_velocity(vb, theta, theta_dot, params):
    """
    计算桨叶法向速度 v_O · ê_θ（Eq.16）
    Drive阶段：此值非零
    Recovery阶段：此值为零
    """
    outer_oar_length = params['l']
    return outer_oar_length * theta_dot + vb * np.cos(theta)


# ============================================================
# 第四步：ODE右端项（Eq.18）
# ============================================================
def compute_dvb_dt(t, vb, cs_leg, cs_trunk, cs_theta, params):
    """
    计算 dv_b/dt（Eq.18右端项）

    直接驱动版本：θ及其导数直接来自实测θ的样条
    这比从x_H/S反推更准确
    """

    # ── 安全检查：vb异常时立即返回 ──
    if not np.isfinite(vb) or abs(vb) > 50.0:
        return 0.0

    mR = params['mR']
    mb = params['mb']
    mO = params['mO']
    C1 = params['C1']
    C2 = params['C2']
    r = params['r']
    d = params['d']

    # 直接从实测θ样条获取（更准确）
    theta = float(cs_theta(t))
    theta_dot = float(cs_theta.derivative(1)(t))
    theta_ddot = float(cs_theta.derivative(2)(t))

    # 协调运动加速度
    xBFdd = float(cs_leg.derivative(2)(t))
    xSBdd = float(cs_trunk.derivative(2)(t))

    # 判断是否Drive（Eq.16）
    v_normal = blade_normal_velocity(vb, theta, theta_dot, params)
    in_drive = abs(v_normal) > 1e-4

    # 船体阻力（Eq.10）
    F_drag = -C1 * vb ** 2

    # 桨叶推力（Eq.11，仅Drive阶段）
    if in_drive:
        F_oar = C2 * v_normal ** 2
    else:
        F_oar = 0.0

    # Eq.18分子
    numerator = (
            F_drag
            + F_oar * np.cos(theta)
            - mR * (xBFdd + r * xSBdd)
            - mO * d * (theta_ddot * np.cos(theta) - theta_dot ** 2 * np.sin(theta))
    )

    return numerator / (mR + mb + mO)


# ============================================================
# 第五步：RK4积分
# ============================================================
def rk4_integrate(cs_leg, cs_trunk, cs_theta, params, t_eval, vb0):
    """
    用RK4积分Eq.18，得到船速时间历程
    """
    N = len(t_eval)
    h = t_eval[1] - t_eval[0]
    vb = vb0
    vb_traj = np.zeros(N)

    for i, t in enumerate(t_eval):
        vb_traj[i] = vb

        # RK4四个斜率
        k1 = compute_dvb_dt(t, vb, cs_leg, cs_trunk, cs_theta, params)
        k2 = compute_dvb_dt(t + h / 2, vb + h / 2 * k1, cs_leg, cs_trunk, cs_theta, params)
        k3 = compute_dvb_dt(t + h / 2, vb + h / 2 * k2, cs_leg, cs_trunk, cs_theta, params)
        k4 = compute_dvb_dt(t + h, vb + h * k3, cs_leg, cs_trunk, cs_theta, params)

        vb = vb + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return vb_traj


# ============================================================
# 第六步：寻找周期性初始速度（Appendix A.4）
# ============================================================
def find_periodic_vb0(cs_leg, cs_trunk, cs_theta, params, t_eval, vb_measured, tol=1e-6):
    """
    用割线法找到满足 v_b(0) = v_b(T) 的初始速度

    直接驱动版本：用实测平均速度作为初始猜测
    """

    def g(vb0):
        traj = rk4_integrate(cs_leg, cs_trunk, cs_theta, params, t_eval, vb0)
        return traj[-1] - vb0

    # 用实测平均速度作为初始猜测
    v1 = np.mean(vb_measured)
    v2 = v1 + 0.1

    print("寻找周期性初始速度...")
    for i in range(20):
        g1, g2 = g(v1), g(v2)
        if abs(g2 - g1) < 1e-12:
            break
        v3 = v2 - g2 * (v2 - v1) / (g2 - g1)
        print(f"  迭代 {i + 1}: vb0 = {v2:.4f} m/s, 误差 = {abs(g2):.2e}")
        if abs(g2) < tol:
            print(f"  收敛！vb0 = {v2:.4f} m/s")
            return v2
        v1, v2 = v2, v3

    return v2


# ============================================================
# 第七步：计算误差J（Eq. A.6）
# ============================================================
def compute_error_j(vb_pred, vb_meas):
    """
    计算船速的误差贡献 E(v_b)

    E = (1/N) Σ (pred - meas)² / Y*²
    其中 Y* = 平均船速
    """
    Y_star = np.mean(vb_meas)
    e = np.mean((vb_pred - vb_meas) ** 2) / Y_star ** 2
    return e


# ============================================================
# 主程序：直接驱动
# ============================================================
def run_direct_drive():
    # --- 读取数据 ---
    vb_meas, F_meas, xBF, xSB, theta_deg, t_common = process_data()

    # --- 单位转换 ---
    theta = np.radians(theta_deg)  # 角度→弧度
    xBF = xBF - xBF[0]  # 转为相对位移
    xSB = xSB - xSB[0]  # 转为相对位移
    T = t_common[-1] - t_common[0]

    print(f"划桨周期 T = {T:.4f} s")
    print(f"实测平均船速 = {np.mean(vb_meas):.3f} m/s")

    # --- 构建样条 ---
    cs_leg, cs_trunk, cs_arm = prepare_splines(xBF, xSB, theta, t_common, PARAMS)
    cs_theta = CubicSpline(t_common, theta)

    # --- 寻找周期性初始速度 ---
    vb0 = find_periodic_vb0(cs_leg, cs_trunk, cs_theta, PARAMS, t_common, vb_meas)

    # --- RK4积分 ---
    vb_pred = rk4_integrate(cs_leg, cs_trunk, cs_theta, PARAMS, t_common, vb0)

    # --- 计算误差 ---
    E_vb = compute_error_j(vb_pred, vb_meas)
    residual = np.mean(np.abs(vb_pred - vb_meas))
    print(f"\n误差 E(v_b) = {E_vb:.8f}")
    print(f"平均残差    = {residual:.4f} m/s")
    print(f"论文trial b参考值：E ≈ 0.00051，残差 ≈ 0.08 m/s")

    # --- 可视化 ---
    plot_results(t_common, vb_pred, vb_meas, theta, cs_theta, cs_leg, cs_trunk, cs_arm, PARAMS)
    plot_results_full(t_common, vb_pred, vb_meas, F_meas, xBF, xSB, theta, cs_theta)

    return vb_pred, E_vb


def plot_results(t, vb_pred, vb_meas, theta, cs_theta, cs_leg, cs_trunk, cs_arm, params):
    """对照论文图4的格式绘图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 船速对比
    axes[0].plot(t, vb_meas, 'k-', linewidth=2, label='实测')
    axes[0].plot(t, vb_pred, 'b--', linewidth=1.5, label='模型预测')
    axes[0].set_ylabel('船速 (m/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 桨角（验证θ导数是否合理）
    theta_rad, theta_dot, theta_ddot = compute_theta_from_splines(t, cs_leg, cs_trunk, cs_arm, params)
    # theta_dot = cs_theta.derivative(1)(t)

    axes[1].plot(t, np.degrees(theta), 'k-', linewidth=2, label='实测桨角')
    axes[1].set_ylabel('桨角 (°)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    ax2 = axes[1].twinx()
    ax2.plot(t, theta_dot, 'r-', linewidth=1, alpha=0.5, label='θ̇（rad/s）')
    ax2.set_ylabel('桨角速度 (rad/s)', color='r')

    axes[-1].set_xlabel('时间 (s)')
    plt.tight_layout()
    plt.savefig('direct_drive_result.png', dpi=150)
    plt.show()


def plot_results_full(t, vb_pred, vb_meas, f_meas, x_bf, x_sb, theta, cs_theta):
    """对照论文图4的完整格式绘图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    # 标记入水和出水时刻
    theta_dot = cs_theta.derivative(1)(t)
    v_normal = (PARAMS['l'] * theta_dot + vb_meas * np.cos(theta))
    drive_mask = np.abs(v_normal) > 1e-4
    transitions = np.diff(drive_mask.astype(int))
    catch_idx = np.where(transitions > 0)[0]
    release_idx = np.where(transitions < 0)[0]

    def mark_events(ax):
        for idx in catch_idx:
            ax.axvline(t[idx], color='b', linestyle='--',
                       linewidth=0.8, alpha=0.6, label='入水')
        for idx in release_idx:
            ax.axvline(t[idx], color='r', linestyle='--',
                       linewidth=0.8, alpha=0.6, label='出水')

    # ── 子图1：船速 ──────────────────────────────
    axes[0].plot(t, vb_meas, 'k-', linewidth=2, label='实测')
    axes[0].plot(t, vb_pred, 'b--', linewidth=1.5, label='模型预测')
    axes[0].set_ylabel('船速 (m/s)')
    axes[0].legend(loc='upper right', fontsize=8)
    mark_events(axes[0])
    axes[0].grid(True, alpha=0.3)

    # ── 子图2：桨柄力 ─────────────────────────────
    axes[1].plot(t, f_meas, 'k-', linewidth=2, label='实测桨柄力')
    axes[1].set_ylabel('桨柄力 (N)')
    axes[1].legend(loc='upper right', fontsize=8)
    mark_events(axes[1])
    axes[1].grid(True, alpha=0.3)

    # ── 子图3：腿位移 ─────────────────────────────
    axes[2].plot(t, x_bf, 'k-', linewidth=2, label='腿位移')
    axes[2].set_ylabel('腿位移 (m)')
    axes[2].legend(loc='upper right', fontsize=8)
    mark_events(axes[2])
    axes[2].grid(True, alpha=0.3)

    # ── 子图4：桨角 ───────────────────────────────
    axes[3].plot(t, np.degrees(theta), 'k-', linewidth=2, label='实测桨角')
    axes[3].set_ylabel('桨角 (°)')
    axes[3].legend(loc='upper right', fontsize=8)
    mark_events(axes[3])
    axes[3].grid(True, alpha=0.3)

    # ── 子图5：背位移 ─────────────────────────────
    axes[4].plot(t, x_sb, 'k-', linewidth=2, label='背位移')
    axes[4].set_ylabel('背位移 (m)')
    axes[4].legend(loc='upper right', fontsize=8)
    mark_events(axes[4])
    axes[4].grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间 (s)')
    plt.tight_layout()
    plt.savefig('direct_drive_result_full.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    vb_predicted, E = run_direct_drive()
