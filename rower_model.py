import numpy as np


class RowerModel:
    """
    Simple rower motion model.

    Generates leg, back, arm motion and resulting handle position.
    """

    def __init__(self, stroke_rate=30):
        self.stroke_rate = stroke_rate
        self.T = 60.0 / stroke_rate

        # motion amplitudes (meters)

        self.leg_amp = 0.6
        self.back_amp = 0.3
        self.arm_amp = 0.4

    # -------------------------------------------------
    # Stroke phase
    # -------------------------------------------------

    def phase(self, t):
        return (t % self.T) / self.T

    # -------------------------------------------------
    # Leg motion
    # -------------------------------------------------

    def leg_motion(self, t):
        p = self.phase(t)

        return self.leg_amp * np.sin(2 * np.pi * p)

    # -------------------------------------------------
    # Back motion
    # -------------------------------------------------

    def back_motion(self, t):
        p = self.phase(t)

        return self.back_amp * np.sin(2 * np.pi * (p - 0.1))

    # -------------------------------------------------
    # Arm motion
    # -------------------------------------------------

    def arm_motion(self, t):
        p = self.phase(t)

        return self.arm_amp * np.sin(2 * np.pi * (p - 0.2))

    # -------------------------------------------------
    # Handle position
    # -------------------------------------------------

    def handle_position(self, t):
        x_BF = self.leg_motion(t)
        x_SB = self.back_motion(t)
        x_HS = self.arm_motion(t)

        x_hand = x_BF + x_SB - x_HS

        return x_hand


if __name__ == '__main__':
    rower = RowerModel()

    t = np.linspace(0, 2, 200)

    x = [rower.handle_position(tt) for tt in t]

    print(x[:10])
