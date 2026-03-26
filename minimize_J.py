import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

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
    # 'C1': 2.80,  # 船体阻力系数
    'C2': 58.7,  # 桨叶阻力系数
    'r': 0.4,  # 质心高度比
    'd': 0.565,  # 桨锁到桨质心距离 (m)
    'IG': 0.85,  # 桨转动惯量 (kg·m²)
    'dLF': 0.0,  # 桨锁前后位置（初始估计）
}

N_KNOTS = 16
# N_KNOTS = 6

# 全局计数器（在文件顶部定义）
_call_count = [0]
_start_time = [None]


# ============================================================
# 核心模块一：由节点值构建样条
# ============================================================
def build_splines(p_vec, t, n_knots=N_KNOTS):
    """
    由优化参数向量构建三条协调运动样条
    """
    leg_knots = p_vec[0: n_knots]
    trunk_knots = p_vec[n_knots: 2 * n_knots]
    arm_knots = p_vec[2 * n_knots: 3 * n_knots]
    d_LF = p_vec[3 * n_knots]

    t_knots = np.linspace(0, t, n_knots, endpoint=False)

    def make_periodic_spline(knots):
        # 首尾闭合以满足周期性
        t_cl = np.append(t_knots, t)
        v_cl = np.append(knots, knots[0])
        return CubicSpline(t_cl, v_cl, bc_type='periodic')

    cs_leg = make_periodic_spline(leg_knots)
    cs_trunk = make_periodic_spline(trunk_knots)
    cs_arm = make_periodic_spline(arm_knots)

    return cs_leg, cs_trunk, cs_arm, d_LF


# ============================================================
# 核心模块二：由样条计算桨角（Eq.8, 9）
# ============================================================
def compute_theta(t, cs_leg, cs_trunk, cs_arm, d_lf, params):
    """
    由协调运动样条通过Eq.8反推桨角θ及其导数

    注意：与直接驱动不同，这里没有实测θ可用
    θ完全由协调运动反推
    """
    s = params['s']

    x_bf = float(cs_leg(t))
    x_sb = float(cs_trunk(t))
    x_hs = float(cs_arm(t))

    x_bf_d = float(cs_leg.derivative(1)(t))
    x_sb_d = float(cs_trunk.derivative(1)(t))
    x_hs_d = float(cs_arm.derivative(1)(t))

    x_bf_dd = float(cs_leg.derivative(2)(t))
    x_sb_dd = float(cs_trunk.derivative(2)(t))
    x_hs_dd = float(cs_arm.derivative(2)(t))

    # Eq.8
    sin_theta = (d_lf + x_hs - x_sb - x_bf) / s
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)

    # Eq.9：θ̇
    if abs(cos_theta) < 1e-6:
        theta_dot = 0.0
    else:
        theta_dot = (x_hs_d - x_bf_d - x_sb_d) / (s * cos_theta)

    # Eq.9：θ̈
    if abs(cos_theta) < 1e-6:
        theta_ddot = 0.0
    else:
        theta_ddot = (((x_hs_dd - x_bf_dd - x_sb_dd) + s * theta_dot ** 2 * np.sin(theta)) / (s * cos_theta))

    return theta, theta_dot, theta_ddot


# ============================================================
# 核心模块三：ODE右端项（Eq.18）
# ============================================================
def compute_dvb_dt(t, vb, cs_leg, cs_trunk, cs_arm, d_lf, params):
    """
    与直接驱动的区别：
    θ来自Eq.8反推，而不是实测θ的样条
    """
    if not np.isfinite(vb) or abs(vb) > 50.0:
        return 0.0

    mr = params['mR']
    mb = params['mb']
    mO = params['mO']
    c1 = params['C1']
    c2 = params['C2']
    r = params['r']
    d = params['d']

    theta, theta_dot, theta_ddot = compute_theta(t, cs_leg, cs_trunk, cs_arm, d_lf, params)

    x_bf_dd = float(cs_leg.derivative(2)(t))
    x_sb_dd = float(cs_trunk.derivative(2)(t))

    # Drive/Recovery：用θ̇符号判断
    v_normal = params['l'] * theta_dot + vb * np.cos(theta)
    # in_drive = theta_dot < 0
    in_drive = v_normal > 1e-6

    F_drag = -c1 * vb ** 2

    if in_drive:
        F_oar = c2 * v_normal ** 2
    else:
        F_oar = 0.0

    numerator = (
            F_drag
            + F_oar * np.cos(theta)
            - mr * (x_bf_dd + r * x_sb_dd)
            - mO * d * (theta_ddot * np.cos(theta) - theta_dot ** 2 * np.sin(theta))
    )

    result = numerator / (mr + mb + mO)
    return result if np.isfinite(result) else 0.0


# ============================================================
# 核心模块四：RK4积分
# ============================================================
def rk4_integrate(cs_leg, cs_trunk, cs_arm, d_lf, params, t_eval, vb0):
    N = len(t_eval)
    h = t_eval[1] - t_eval[0]
    vb = vb0
    vb_traj = np.zeros(N)

    for i, t in enumerate(t_eval):
        vb_traj[i] = vb

        k1 = compute_dvb_dt(t, vb, cs_leg, cs_trunk, cs_arm, d_lf, params)
        k2 = compute_dvb_dt(t + h / 2, vb + h / 2 * k1, cs_leg, cs_trunk, cs_arm, d_lf, params)
        k3 = compute_dvb_dt(t + h / 2, vb + h / 2 * k2, cs_leg, cs_trunk, cs_arm, d_lf, params)
        k4 = compute_dvb_dt(t + h, vb + h * k3, cs_leg, cs_trunk, cs_arm, d_lf, params)

        vb = vb + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return vb_traj


# ============================================================
# 核心模块五：割线法求周期性初始速度
# ============================================================
def find_periodic_vb0(cs_leg, cs_trunk, cs_arm, d_lf, params, t_eval, vb_init, tol=1e-6):
    def g(vb0):
        traj = rk4_integrate(cs_leg, cs_trunk, cs_arm, d_lf, params, t_eval, vb0)
        return traj[-1] - vb0

    v1, v2 = vb_init, vb_init + 0.1

    for _ in range(20):
        g1, g2 = g(v1), g(v2)
        if not np.isfinite(g1) or not np.isfinite(g2):
            return vb_init  # 积分发散时返回初始猜测
        if abs(g2 - g1) < 1e-12:
            break
        v3 = v2 - g2 * (v2 - v1) / (g2 - g1)
        if not np.isfinite(v3) or abs(v3) > 20.0:
            v3 = vb_init
        if abs(g2) < tol:
            return v2
        v1, v2 = v2, v3

    return v2


# ============================================================
# 核心模块六：误差函数J（Eq. A.6）
# ============================================================
def compute_j(p_vec, t_eval, measured, params, fit_vars, char_values, t):
    """计算误差 J"""

    # ── 进度监控 ──
    _call_count[0] += 1
    if _start_time[0] is None:
        _start_time[0] = time.time()
    call_start = time.time()

    try:
        cs_leg, cs_trunk, cs_arm, d_lf = build_splines(p_vec, t)

        # 寻找周期性初始速度
        vb0 = find_periodic_vb0(cs_leg, cs_trunk, cs_arm, d_lf, params, t_eval, vb_init=np.mean(measured['vb']))

        # RK4积分得到预测船速
        vb_pred = rk4_integrate(cs_leg, cs_trunk, cs_arm, d_lf, params, t_eval, vb0)

        total_J = 0.0
        for var in fit_vars:
            if var == 'vb':
                predicted = vb_pred
                measured_ = measured['vb']
            elif var == 'theta':
                # 从样条重建θ时间历程
                predicted = np.array([compute_theta(t, cs_leg, cs_trunk, cs_arm, d_lf, params)[0] for t in t_eval])
                measured_ = measured['theta']
            elif var == 'xBF':
                predicted = cs_leg(t_eval)
                measured_ = measured['xBF']
            elif var == 'xSB':
                predicted = cs_trunk(t_eval)
                measured_ = measured['xSB']
            else:
                continue

            E_j = (np.mean((predicted - measured_) ** 2) / char_values[var] ** 2)
            total_J += E_j

        result_J = total_J / len(fit_vars)

        # ── 每10次调用打印一次 ──
        if _call_count[0] % 10 == 0:
            elapsed = time.time() - _start_time[0]
            per_call = time.time() - call_start
            print(f"  调用 {_call_count[0]:4d} | "
                  f"J = {result_J:.6f} | "
                  f"vb0 = {vb0:.3f} m/s | "
                  f"单次耗时 {per_call:.2f}s | "
                  f"累计 {elapsed:.0f}s")

        return result_J

    except Exception:
        return 1e6  # 数值错误时返回大值


# ============================================================
# 核心模块七：初始参数猜测
# ============================================================
def init_p0(measured, t, params, n_knots=N_KNOTS):
    """
    用实测数据插值到节点，作为优化起点
    这比随机初始化收敛快得多
    """
    t = measured['t']
    t_k = np.linspace(0, t, n_knots, endpoint=False)

    # ── 诊断打印 ──
    print(f"n_knots = {n_knots}")
    print(f"t 长度  = {len(t)}")
    print(f"t_k 形状 = {t_k.shape}")

    # 腿和躯干：直接从实测数据插值
    leg_k = CubicSpline(t, measured['xBF'])(t_k)
    trunk_k = CubicSpline(t, measured['xSB'])(t_k)

    # 手臂：由Eq.8从实测θ反推
    s = params['s']
    dLF = params['dLF']
    xHS = (s * np.sin(measured['theta']) - dLF + measured['xSB'] + measured['xBF'])
    arm_k = CubicSpline(t, xHS)(t_k)

    # ✅ 修复：确保所有数组是1维
    leg_k = np.asarray(leg_k).flatten()
    trunk_k = np.asarray(trunk_k).flatten()
    arm_k = np.asarray(arm_k).flatten()

    p0 = np.concatenate([leg_k, trunk_k, arm_k, [dLF]])
    print(f"p0 shape: {p0.shape}（预期：({3 * n_knots + 1}) = ({3 * N_KNOTS + 1})）")
    return p0


# ============================================================
# 主程序：最小化J
# ============================================================
def run_minimize_j():
    # --- 读取并预处理数据 ---
    vb_meas, F_meas, xBF, xSB, theta_deg, t_common = process_data(50)

    theta = np.radians(theta_deg)
    xBF = xBF - xBF[0]
    xSB = xSB - xSB[0]
    T = t_common[-1] - t_common[0]

    measured = {
        't': t_common,
        'vb': vb_meas,
        'xBF': xBF,
        'xSB': xSB,
        'theta': theta,
    }

    # 特征值 Y*（用于归一化误差）
    char_values = {
        'vb': np.mean(vb_meas),
        'theta': np.ptp(theta),
        'xBF': np.ptp(xBF),
        'xSB': np.ptp(xSB),
    }

    # --- 初始参数 ---
    p0 = init_p0(measured, T, PARAMS)
    print(f"优化参数数量：{len(p0)}")
    print(f"初始J（用实测数据插值）：", end=' ')
    J0 = compute_j(
        p0, t_common, measured, PARAMS,
        fit_vars=['vb', 'theta', 'xBF', 'xSB'],
        char_values=char_values, t=T)
    print(f"{J0:.6f}")

    # ── 优化前：先测一次单次调用耗时 ──
    print("预热测试（测量单次J计算耗时）...")
    t_test = time.time()
    J_test = compute_j(
        p0, t_common, measured, PARAMS,
        fit_vars=['theta', 'xBF', 'xSB'],
        char_values=char_values, t=T)
    t_single = time.time() - t_test
    print(f"单次J计算耗时：{t_single:.2f} s")
    print(f"初始 J = {J_test:.6f}")

    # L-BFGS-B 通常需要 500~2000 次J计算
    est_min = t_single * 1000 / 60
    est_max = t_single * 3000 / 60
    print(f"预估总时间：{est_min:.0f} ~ {est_max:.0f} 分钟")
    print(f"（L-BFGS-B 通常需要 1000~3000 次J计算）\n")

    # ── 重置计数器 ──
    _call_count[0] = 0
    _start_time[0] = None

    # --- 优化 ---
    iteration = [0]
    j_history = []

    def callback(p_vec):
        iteration[0] += 1
        J = compute_j(p_vec, t_common, measured, PARAMS,
                      fit_vars=['vb', 'theta', 'xBF', 'xSB'],
                      char_values=char_values, t=T)
        j_history.append(J)
        if iteration[0] % 10 == 0:
            print(f"  迭代 {iteration[0]:4d}:  J = {J:.8f}")

    print("开始优化...")
    opt_start = time.time()
    result = minimize(
        compute_j,
        p0,
        args=(
            t_common, measured, PARAMS,
            ['vb', 'theta', 'xBF', 'xSB'],  # 拟合的参数
            char_values, T
        ),
        method='L-BFGS-B',
        # method='Nelder-Mead',
        callback=callback,
        options={'maxiter': 300, 'ftol': 1e-10}
    )

    opt_total = time.time() - opt_start
    print(f"\n优化完成！")
    print(f"总耗时：{opt_total / 60:.1f} 分钟")
    print(f"总调用次数：{_call_count[0]}")
    print(f"J_min = {result.fun:.8f}")
    print(f"收敛状态：{result.message}")

    # --- 用最优参数做最终预测 ---
    p_opt = result.x
    cs_leg, cs_trunk, cs_arm, d_LF = build_splines(p_opt, T)

    vb0 = find_periodic_vb0(cs_leg, cs_trunk, cs_arm, d_LF, PARAMS, t_common, np.mean(vb_meas))
    vb_pred = rk4_integrate(cs_leg, cs_trunk, cs_arm, d_LF, PARAMS, t_common, vb0)

    # --- 计算最终误差 ---
    Y_star = np.mean(vb_meas)
    E_vb = np.mean((vb_pred - vb_meas) ** 2) / Y_star ** 2
    residual = np.mean(np.abs(vb_pred - vb_meas))
    print(f"E(v_b)  = {E_vb:.8f}")
    print(f"平均残差 = {residual:.4f} m/s")
    print(f"论文trial b参考：E ≈ 0.00051，残差 ≈ 0.08 m/s")

    # --- 绘图 ---
    plot_results(t_common, vb_pred, vb_meas, xBF, xSB, theta, cs_leg, cs_trunk, cs_arm, j_history)

    return vb_pred, p_opt, j_history


# ============================================================
# 绘图
# ============================================================
def plot_results(t, vb_pred, vb_meas, x_bf, x_sb, theta, cs_leg, cs_trunk, cs_arm, j_history):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 船速
    axes[0].plot(t, vb_meas, 'k-', lw=2, label='实测')
    axes[0].plot(t, vb_pred, 'b--', lw=1.5, label='模型预测')
    axes[0].set_ylabel('船速 (m/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 腿位移
    axes[1].plot(t, x_bf, 'k-', lw=2, label='实测')
    axes[1].plot(t, cs_leg(t), 'b--', lw=1.5, label='模型')
    axes[1].set_ylabel('腿位移 (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 背位移
    axes[2].plot(t, x_sb, 'k-', lw=2, label='实测')
    axes[2].plot(t, cs_trunk(t), 'b--', lw=1.5, label='模型')
    axes[2].set_ylabel('背位移 (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 桨角
    theta_pred = np.array([compute_theta(ti, cs_leg, cs_trunk, cs_arm, 0.0, PARAMS)[0] for ti in t])
    axes[3].plot(t, np.degrees(theta), 'k-', lw=2, label='实测')
    axes[3].plot(t, np.degrees(theta_pred), 'b--', lw=1.5, label='模型')
    axes[3].set_ylabel('桨角 (°)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间 (s)')
    plt.tight_layout()
    plt.savefig('minimize_J_result.png', dpi=150)
    plt.show()

    # J收敛曲线
    if j_history:
        plt.figure(figsize=(8, 4))
        plt.plot(j_history)
        plt.xlabel('迭代次数')
        plt.ylabel('误差 J')
        plt.title('优化收敛过程')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('J_convergence.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    vb_predicted, p_optimize, J_history = run_minimize_j()
