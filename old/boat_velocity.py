import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# ============================================================
# 1. 模型常数（单人艇）
# ============================================================
params = {
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
    'dLF': 0.0,  # 桨锁前后位置（待拟合）
}


# ============================================================
# 2. 构建协调运动样条（实际使用时从数据拟合得到）
# ============================================================
def build_coordination_splines(knot_ordinates, t, n=16):
    """
    用周期性三次样条构建协调运动函数
    knot_ordinates: shape (3, n)，分别对应xBF, xSB, xHS
    返回三个CubicSpline对象及其导数
    """
    t_knots = np.linspace(0, t, n + 1)
    knot_ordinates = np.hstack([knot_ordinates, knot_ordinates[:, 0:1]])

    splines = {}
    for i, name in enumerate(['xBF', 'xSB', 'xHS']):
        # 周期性边界条件
        cs = CubicSpline(
            t_knots,
            knot_ordinates[i],
            bc_type='periodic'
        )
        splines[name] = cs  # 位置
        splines[name + 'd'] = cs.derivative(1)  # 速度
        splines[name + 'dd'] = cs.derivative(2)  # 加速度

    return splines


# ============================================================
# 3. 由协调运动计算桨角θ及其导数
# ============================================================
def compute_oar_angle(t, splines):
    """由Eq.8计算θ，由Eq.9计算θ̇和θ̈"""
    s = params['s']
    dLF = params['dLF']

    xBF = splines['xBF'](t)
    xSB = splines['xSB'](t)
    xHS = splines['xHS'](t)

    xBFd = splines['xBFd'](t)
    xSBd = splines['xSBd'](t)
    xHSd = splines['xHSd'](t)

    xBFdd = splines['xBFdd'](t)
    xSBdd = splines['xSBdd'](t)
    xHSdd = splines['xHSdd'](t)

    # Eq.8: θ
    sin_theta = (dLF + xHS - xSB - xBF) / s
    sin_theta = np.clip(sin_theta, -1, 1)  # 防止数值误差
    theta = np.arcsin(sin_theta)

    # Eq.9: θ̇
    # cos_theta = np.clip(np.cos(theta), 1e-6, None)
    cos_theta = np.cos(theta)
    cos_theta = np.sign(cos_theta) * np.maximum(abs(cos_theta), 1e-6)
    theta_dot = (xHSd - xBFd - xSBd) / (s * cos_theta)

    # Eq.9: θ̈
    theta_ddot = ((xHSdd - xBFdd - xSBdd) + s * theta_dot ** 2 * np.sin(theta)) / (s * cos_theta)

    return theta, theta_dot, theta_ddot


# ============================================================
# 4. 判断Drive/Recovery阶段
# ============================================================
def blade_normal_velocity(vb, theta, theta_dot):
    """计算桨叶法向速度 v_O · ê_θ"""
    param_l = params['l']
    return param_l * theta_dot + vb * np.cos(theta)


def is_drive(vb, theta, theta_dot):
    """判断是否处于Drive阶段"""
    v_normal = blade_normal_velocity(vb, theta, theta_dot)
    return v_normal != 0  # 实际实现需要追踪状态


# ============================================================
# 5. 右端项：dv_b/dt
# ============================================================
def dvb_dt(t, vb, splines, drive_phase):
    """
    计算 dv_b/dt（Eq.18的右端项）
    drive_phase: bool，当前是否为发力阶段
    """
    mR = params['mR']
    mb = params['mb']
    mO = params['mO']
    C1 = params['C1']
    C2 = params['C2']
    r = params['r']
    d = params['d']
    param_l = params['l']

    # 从样条获取协调运动加速度
    xBFdd = splines['xBFdd'](t)
    xSBdd = splines['xSBdd'](t)

    # 计算桨角及其导数
    theta, theta_dot, theta_ddot = compute_oar_angle(t, splines)

    # 船体阻力（向后为负）
    F_drag = -C1 * vb * abs(vb)

    # 桨叶推力（仅Drive阶段）
    if drive_phase:
        v_normal = param_l * theta_dot + vb * np.cos(theta)
        F_oar = -C2 * v_normal * abs(v_normal)
    else:
        F_oar = 0.0

    # 各项贡献
    numerator = (
            F_drag
            + F_oar * np.cos(theta)
            - mR * (xBFdd + r * xSBdd)
            - mO * d * (theta_ddot * np.cos(theta) - theta_dot ** 2 * np.sin(theta))
    )
    denominator = mR + mb + mO

    return numerator / denominator


# ============================================================
# 6. 求解ODE（单次积分）
# ============================================================
def integrate_stroke(vb0, splines):
    """
    对一个完整划桨周期进行数值积分
    返回时间序列和船速序列
    """
    param_t = params['T']
    n = 50
    dt = param_t / (2 * n)
    t_eval = np.linspace(0, param_t, n + 1)

    def rhs(t, y):
        vb = y[0]
        theta, theta_dot, _ = compute_oar_angle(t, splines)
        v_norm = blade_normal_velocity(vb, theta, theta_dot)
        in_drive = v_norm > 0
        return [dvb_dt(t, vb, splines, in_drive)]

    sol = solve_ivp(
        rhs,
        [0, param_t],
        [vb0],
        method='RK45',
        t_eval=t_eval,
        max_step=dt,
        rtol=1e-6,
        atol=1e-8
    )

    return sol.t, sol.y[0]


# ============================================================
# 7. 迭代求周期性稳态（割线法）
# ============================================================
def find_periodic_solution(splines, vb_init=4.0):
    """
    用割线法找到满足 v_b(0) = v_b(T) 的初始速度
    对应论文Appendix A.4
    """

    def g(vb0):
        """g(v) = v_b(T) - v_b(0)"""
        _, vb_series = integrate_stroke(vb0, splines)
        return vb_series[-1] - vb0

    # 割线法迭代
    v1 = vb_init
    v2 = vb_init + 0.1

    for iteration in range(20):
        g1 = g(v1)
        g2 = g(v2)

        if abs(g2 - g1) < 1e-10:
            break

        # 割线法更新
        v3 = v2 - g2 * (v2 - v1) / (g2 - g1)
        if abs(v3 - v2) < 1e-6:  # 收敛判断
            print(f"收敛于第 {iteration + 1} 次迭代")
            break

        v1, v2 = v2, v3
        print(f"迭代 {iteration + 1}: v_b(0) = {v2:.6f} m/s, "
              f"误差 = {abs(g2):.2e}")

    return v2


# ============================================================
# 8. 主程序示例
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = params['T']
    N = 16  # 样条节点数

    # 用示例协调运动（实际应从数据拟合）
    T_Knots = np.linspace(0, T, N, endpoint=False)
    Knot_Ordinates = np.array([
        0.3 * np.sin(2 * np.pi * T_Knots / T),  # xBF
        0.15 * np.sin(2 * np.pi * T_Knots / T),  # xSB
        0.1 * np.sin(2 * np.pi * T_Knots / T),  # xHS
    ])

    Splines = build_coordination_splines(Knot_Ordinates, T, N)

    # 求周期性稳态
    vb_periodic = find_periodic_solution(Splines)

    # 最终积分并绘图
    t_sol, vb_sol = integrate_stroke(vb_periodic, Splines)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(8, 4))
    plt.plot(t_sol, vb_sol)
    plt.xlabel('时间 t (s)')
    plt.ylabel('船速 v_b (m/s)')
    plt.title('船速时间历程')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('boat_velocity.png', dpi=150)
    plt.show()
