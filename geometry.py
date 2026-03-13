import numpy as np


class OarGeometry:
    """
    Geometry model of the rowing oar system.

    Coordinate system:
    - origin: oarlock (桨锁)
    - +x: boat forward direction
    - +y: outward from boat

    Angle theta:
    - measured from +x-axis
    - positive = oar pointing outward
    """

    def __init__(
            self,
            l_in: float,
            l_out: float,
            oarlock_height: float = 0.0,
            blade_area: float = 0.05,
    ):
        """
        Parameters
        ----------
        l_in : float
            Length from oarlock to handle (m)
        l_out : float
            Length from oarlock to blade (m)
        oarlock_height : float
            Height of oarlock above water (m)
        blade_area : float
            Area of blade (m^2)
        """

        self.l_in = l_in
        self.l_out = l_out
        self.l_total = l_in + l_out

        self.oarlock_height = oarlock_height
        self.blade_area = blade_area

    # -------------------------------------------------
    # Basic geometry
    # -------------------------------------------------

    def blade_position(self, theta: float):
        """
        Position of blade relative to oarlock.

        Parameters
        ----------
        theta : float
            Oar angle (radians)

        Returns
        -------
        (x, y)
        """

        x = self.l_out * np.cos(theta)
        y = self.l_out * np.sin(theta)

        return np.array([x, y])

    def handle_position(self, theta: float):
        """
        Position of handle (rower's hand).

        Returns
        -------
        (x, y)
        """

        x = -self.l_in * np.cos(theta)
        y = -self.l_in * np.sin(theta)

        return np.array([x, y])

    # -------------------------------------------------
    # Lever mechanics
    # -------------------------------------------------

    def mechanical_ratio(self):
        """
        Lever ratio of the oar.

        Returns
        -------
        ratio = L_out / L_in
        """

        return self.l_out / self.l_in

    def water_force_from_handle_force(self, f_handle: float):
        """
        Convert handle force to blade force.

        F_water = (L_in / L_out) * F_handle
        """

        return (self.l_in / self.l_out) * f_handle

    def handle_force_from_water_force(self, f_water: float):
        """
        Convert water force to handle force.
        """

        return (self.l_out / self.l_in) * f_water

    # -------------------------------------------------
    # Blade immersion
    # -------------------------------------------------

    def blade_depth(self, theta: float):
        """
        Estimate blade depth relative to water surface.
        """

        blade_y = self.blade_position(theta)[1]

        depth = blade_y - self.oarlock_height

        return depth

    def blade_in_water(self, theta: float):
        """
        Check if blade is in water.
        """

        return self.blade_depth(theta) < 0
