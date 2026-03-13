import numpy as np


class OarKinematics:
    """
    Oar kinematics driven by rower handle motion.
    """

    def __init__(self, geometry, rower_model):
        self.geometry = geometry
        self.rower = rower_model

        self.L_in = geometry.l_in

    # -------------------------------------------------
    # Solve oar angle
    # -------------------------------------------------

    def theta(self, t):
        """
        Compute oar angle from handle position.
        """

        x_hand = self.rower.handle_position(t)

        ratio = -x_hand / self.L_in

        ratio = np.clip(ratio, -1.0, 1.0)

        theta = np.arccos(ratio)

        return theta

    # -------------------------------------------------
    # Angular velocity
    # -------------------------------------------------

    def omega(self, t, dt=1e-4):
        """
        Numerical derivative of theta.
        """

        theta1 = self.theta(t - dt)
        theta2 = self.theta(t + dt)

        return (theta2 - theta1) / (2 * dt)

    # -------------------------------------------------
    # Blade velocity
    # -------------------------------------------------

    def blade_velocity(self, t):
        """
        Velocity of blade tip.
        """

        theta = self.theta(t)
        omega = self.omega(t)

        L = self.geometry.l_out

        vx = -L * omega * np.sin(theta)
        vy = L * omega * np.cos(theta)

        return np.array([vx, vy])
