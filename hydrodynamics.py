import numpy as np


class BladeHydrodynamics:
    """
    Hydrodynamic model of rowing blade.
    """

    def __init__(
            self,
            geometry,
            rho=1000.0,
            cd=1.8
    ):
        """
        Parameters
        ----------
        geometry : OarGeometry
        rho : float
            water density (kg/m^3)
        cd : float
            drag coefficient of blade
        """

        self.geometry = geometry
        self.rho = rho
        self.Cd = cd

    # -------------------------------------------------
    # Relative velocity
    # -------------------------------------------------

    def relative_velocity(self, blade_velocity, boat_velocity):
        """
        Velocity of blade relative to water.
        """

        v_boat = np.array([boat_velocity, 0.0])

        return blade_velocity - v_boat

    # -------------------------------------------------
    # Hydrodynamic force
    # -------------------------------------------------

    def blade_force(self, blade_velocity, boat_velocity):
        """
        Force exerted by water on blade.
        """

        v_rel = self.relative_velocity(blade_velocity, boat_velocity)

        v = np.linalg.norm(v_rel)

        if v == 0:
            return np.zeros(2)

        A = self.geometry.blade_area

        F_mag = 0.5 * self.rho * self.Cd * A * v ** 2

        direction = -v_rel / v

        return F_mag * direction

    # -------------------------------------------------
    # Boat thrust
    # -------------------------------------------------

    def boat_thrust(self, blade_force):
        """
        Extract propulsion force acting on boat.
        """

        # Only x-direction propels the boat
        return blade_force[0]
