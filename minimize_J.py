import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from data_process import process_data


class RowingSimulator:
    """
    基于论文Cabrera et al. (2006)的划桨模型
    通过最小化误差J来拟合协调运动
    """

    def __init__(self, params, measured_data):
        """
        Parameters
        ----------
        params : dict
            模型物理常数（来自Table 1）
        measured_data : dict
            实测数据，键为变量名，值为numpy数组
        """
        self.p = params
        self.data = measured_data
        self.T = measured_data['t'][-1] - measured_data['t'][0]
        self.N = len(measured_data['t'])
        self.n_knots = 16

        # 特征值Y*（用于归一化误差）
        self.char_values = {
            'vb': np.mean(measured_data['vb']),
            'F': np.max(measured_data['F']),
            'xBF': np.ptp(measured_data['xBF']),
            'xSB': np.ptp(measured_data['xSB']),
            'theta': np.ptp(measured_data['theta']),
        }

    # ----------------------------------------------------------
    # 1. 由节点值构建样条
    # ----------------------------------------------------------

    def build_splines(self, p_vec):
        """由优化参数向量构建三条协调运动样条"""
        n = self.n_knots
        t_knots = np.linspace(0, self.T, n, endpoint=False)

        # 解包参数向量
        leg_knots = p_vec[0:n]
        trunk_knots = p_vec[n:2 * n]
        arm_knots = p_vec[2 * n:3 * n]
        d_LF = p_vec[3 * n]

        # 构建周期性三次样条
        def make_spline(knots):
            t_cl = np.append(t_knots, self.T)
            v_cl = np.append(knots, knots[0])
            return CubicSpline(t_cl, v_cl, bc_type='periodic')

        cs_leg = make_spline(leg_knots)
        cs_trunk = make_spline(trunk_knots)
        cs_arm = make_spline(arm_knots)

        return cs_leg, cs_trunk, cs_arm, d_LF

    # ----------------------------------------------------------
    # 2. 计算桨角及其导数（Eq.8, 9）
    # ----------------------------------------------------------

    def compute_theta(self, cs_leg, cs_trunk, cs_arm, d_LF, t):
        """由协调运动计算桨角θ及其一阶、二阶导数"""
        s = self.p['s']

        xBF = cs_leg(t)
        xBFd = cs_leg.derivative(1)(t)
        xSB = cs_trunk(t)
        xSBd = cs_trunk.derivative(1)(t)
        xHS = cs_arm(t)
        xHSd = cs_arm.derivative(1)(t)

        xBFdd = cs_leg.derivative(2)(t)
        xSBdd = cs_trunk.derivative(2)(t)
        xHSdd = cs_arm.derivative(2)(t)

        # Eq.8：sinθ = (d_LF + x_HS - x_SB - x_BF) / s
        sin_theta = (d_LF + xHS - xSB - xBF) / s
        sin_theta = np.clip(sin_theta, -1, 1)
        theta = np.arcsin(sin_theta)

        # Eq.9：θ̇
        cos_theta = np.cos(theta)
        theta_dot = (xHSd - xBFd - xSBd) / (s * cos_theta)

        # Eq.9：θ̈
        theta_ddot = ((xHSdd - xBFdd - xSBdd) + s * theta_dot ** 2 * np.sin(theta)) / (s * cos_theta)

        return theta, theta_dot, theta_ddot

    # ----------------------------------------------------------
    # 3. ODE右端项（Eq.18）
    # ----------------------------------------------------------

    def oar_force(self, vb, theta, theta_dot, in_drive):
        """计算桨叶推力（Model 1，Eq.11）"""
        if not in_drive:
            return 0.0
        L = self.p['l']
        C2 = self.p['C2']
        v_normal = L * theta_dot + vb * np.cos(theta)
        return C2 * v_normal ** 2

    def dvb_dt(self, t, vb, cs_leg, cs_trunk, cs_arm, d_lf, in_drive):
        """Eq.18的右端项"""
        mR = self.p['mR']
        mb = self.p['mb']
        mO = self.p['mO']
        C1 = self.p['C1']
        r = self.p['r']
        d = self.p['d']

        theta, theta_dot, theta_ddot = self.compute_theta(cs_leg, cs_trunk, cs_arm, d_lf, t)

        xBFdd = cs_leg.derivative(2)(t)
        xSBdd = cs_trunk.derivative(2)(t)

        F_drag = -C1 * vb ** 2
        F_oar = self.oar_force(vb, theta, theta_dot, in_drive)

        numerator = (
                F_drag
                + F_oar * np.cos(theta)
                - mR * (xBFdd + r * xSBdd)
                - mO * d * (theta_ddot * np.cos(theta) - theta_dot ** 2 * np.sin(theta))
        )
        return numerator / (mR + mb + mO)

    # ----------------------------------------------------------
    # 4. RK4积分
    # ----------------------------------------------------------

    def integrate_rk4(self, p_vec):
        """
        用RK4积分Eq.18，返回船速时间历程
        同时处理Drive/Recovery切换
        """
        cs_leg, cs_trunk, cs_arm, d_LF = self.build_splines(p_vec)
        t_eval = np.linspace(0, self.T, self.N)
        h = t_eval[1] - t_eval[0]

        # 寻找周期性初始速度（割线法）
        vb0 = self._find_periodic_vb0(cs_leg, cs_trunk, cs_arm, d_LF)

        vb_traj = np.zeros(self.N)
        vb = vb0
        in_drive = False

        for i, t in enumerate(t_eval):
            vb_traj[i] = vb

            # 判断Drive/Recovery（Eq.16）
            _, theta_dot, _ = self.compute_theta(cs_leg, cs_trunk, cs_arm, d_LF, t)
            theta, _, _ = self.compute_theta(cs_leg, cs_trunk, cs_arm, d_LF, t)
            v_normal = (self.p['l'] * theta_dot + vb * np.cos(theta))
            in_drive = abs(v_normal) > 1e-6

            # RK4四步
            def f(t_, v_):
                return self.dvb_dt(t_, v_, cs_leg, cs_trunk, cs_arm, d_LF, in_drive)

            k1 = f(t, vb)
            k2 = f(t + h / 2, vb + h / 2 * k1)
            k3 = f(t + h / 2, vb + h / 2 * k2)
            k4 = f(t + h, vb + h * k3)

            vb = vb + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return vb_traj

    def _find_periodic_vb0(self, cs_leg, cs_trunk, cs_arm, d_lf, tol=1e-6):
        """割线法寻找周期性初始船速（Appendix A.4）"""

        def g(vb0):
            # 简化版：只积分一个周期，返回终末速度-初始速度
            p_tmp = np.zeros(3 * self.n_knots + 1)
            # 此处传入完整p_vec（简化示意）
            vb_traj = self._single_integrate(vb0, cs_leg, cs_trunk, cs_arm, d_lf)
            return vb_traj[-1] - vb0

        v1, v2 = 4.0, 4.1
        for _ in range(20):
            g1, g2 = g(v1), g(v2)
            if abs(g2 - g1) < 1e-12:
                break
            v3 = v2 - g2 * (v2 - v1) / (g2 - g1)
            if abs(g2) < tol:
                return v2
            v1, v2 = v2, v3
        return v2

    def _single_integrate(self, vb0, cs_leg, cs_trunk, cs_arm, d_lf):
        """单次RK4积分（供割线法调用）"""
        t_eval = np.linspace(0, self.T, self.N)
        h = t_eval[1] - t_eval[0]
        vb = vb0
        vb_traj = []

        for t in t_eval:
            vb_traj.append(vb)
            theta, theta_dot, _ = self.compute_theta(cs_leg, cs_trunk, cs_arm, d_lf, t)
            v_normal = (self.p['l'] * theta_dot + vb * np.cos(theta))
            in_drive = abs(v_normal) > 1e-6

            def f(t_, v_):
                return self.dvb_dt(t_, v_, cs_leg, cs_trunk, cs_arm, d_lf, in_drive)

            k1 = f(t, vb)
            k2 = f(t + h / 2, vb + h / 2 * k1)
            k3 = f(t + h / 2, vb + h / 2 * k2)
            k4 = f(t + h, vb + h * k3)
            vb = vb + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return np.array(vb_traj)

    # ----------------------------------------------------------
    # 5. 误差函数J（Eq. A.6）
    # ----------------------------------------------------------

    def compute_j(self, p_vec, fit_vars):
        """
        计算误差J

        Parameters
        ----------
        p_vec : 优化参数向量
        fit_vars : list，参与拟合的变量名
                   可选：'vb', 'F', 'xBF', 'xSB', 'theta'
        """
        try:
            # 运行模型
            vb_pred = self.integrate_rk4(p_vec)

            total_J = 0.0
            n_vars = len(fit_vars)

            for var in fit_vars:
                measured = self.data[var]
                char_val = self.char_values[var]

                if var == 'vb':
                    predicted = vb_pred
                elif var == 'theta':
                    # 从样条重建预测桨角
                    cs_leg, cs_trunk, cs_arm, d_LF = self.build_splines(p_vec)
                    t_eval = np.linspace(0, self.T, self.N)
                    predicted, _, _ = self.compute_theta(cs_leg, cs_trunk, cs_arm, d_LF, t_eval)
                # F_hand等其他变量需要矩阵求逆（Eq.17）
                # 此处暂时省略，后续补充

                E_j = (np.mean((predicted - measured) ** 2) / char_val ** 2)
                total_J += E_j

            return total_J / n_vars

        except Exception:
            return 1e6  # 数值错误时返回大值

    # ----------------------------------------------------------
    # 6. 优化入口
    # ----------------------------------------------------------

    def fit(self, fit_vars=None, nm=5):
        """
        最小化误差J，寻找最优协调运动节点值

        Parameters
        ----------
        fit_vars : 参与拟合的变量，默认全部5个
        nm : 拟合变量数量
        """
        if fit_vars is None:
            fit_vars = ['vb', 'F', 'xBF', 'xSB', 'theta']

        # 初始猜测：用实测数据插值到节点
        p0 = self._init_from_measured()

        print(f"开始优化，拟合变量：{fit_vars}")
        print(f"优化参数数量：{len(p0)}")

        iteration = [0]
        J_history = []

        def callback(p_vec):
            iteration[0] += 1
            J = self.compute_j(p_vec, fit_vars)
            J_history.append(J)
            if iteration[0] % 10 == 0:
                print(f"  迭代 {iteration[0]:4d}:  J = {J:.8f}")

        result = minimize(
            self.compute_j,
            p0,
            args=(fit_vars,),
            method='L-BFGS-B',  # 论文使用BFGS
            callback=callback,
            options={
                'maxiter': 500,
                'ftol': 1e-10,
                'gtol': 1e-8,
            }
        )

        print(f"\n优化完成：J_min = {result.fun:.8f}")
        print(f"对照论文trial a参考值：J ≈ 0.00024")

        return result.x, J_history

    def _init_from_measured(self):
        """用实测数据插值生成初始节点值"""
        n = self.n_knots
        t_k = np.linspace(0, self.T, n, endpoint=False)
        t = self.data['t']

        leg_k = CubicSpline(t, self.data['xBF'])(t_k)
        trunk_k = CubicSpline(t, self.data['xSB'])(t_k)

        # x_H/S 由Eq.8反推
        s = self.p['s']
        dLF = 0.0
        xHS = (s * np.sin(self.data['theta'])
               - dLF
               + self.data['xSB']
               + self.data['xBF'])
        arm_k = CubicSpline(t, xHS)(t_k)

        p0 = np.concatenate([leg_k, trunk_k, arm_k, [dLF]])
        return p0


if __name__ == '__main__':
    # 读取实测数据
    vb, F, xBF, xSB, theta_deg, t = process_data()
    theta = np.radians(theta_deg)

    # 打包数据
    measured = {
        't': t, 'vb': vb, 'F': F,
        'xBF': xBF, 'xSB': xSB, 'theta': theta
    }

    # 模型常数（Table 1，单人艇）
    params = {
        'T': 1.94, 'mR': 75.0, 'mb': 19.7, 'mO': 1.2,
        's': 0.83, 'l': 1.805, 'C1': 3.16,
        'C2': 58.7, 'r': 0.4, 'd': 0.565,
    }

    # 创建模拟器并优化
    sim = RowingSimulator(params, measured)

    # 对应论文trial a：拟合全部5个变量
    p_opt, J_history = sim.fit(fit_vars=['vb', 'F', 'xBF', 'xSB', 'theta'])
