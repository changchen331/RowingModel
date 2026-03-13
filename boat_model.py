class BoatModel:
    """
    1D boat dynamics model.
    """

    def __init__(
            self,
            mass=90.0,
            drag_coeff=30.0
    ):
        """
        Parameters
        ----------
        mass : float
            mass of boat + rower (kg)
        drag_coeff : float
            quadratic drag coefficient
        """

        self.mass = mass
        self.drag_coeff = drag_coeff

    # -------------------------------------------------
    # Drag force
    # -------------------------------------------------

    def drag_force(self, velocity):
        """
        Hydrodynamic drag on the boat.

        F_drag = k v^2
        """

        return self.drag_coeff * velocity ** 2

    # -------------------------------------------------
    # Boat acceleration
    # -------------------------------------------------

    def acceleration(self, thrust, velocity):
        """
        Compute dv/dt.

        m dv/dt = thrust - drag
        """

        drag = self.drag_force(velocity)

        net_force = thrust - drag

        return net_force / self.mass

    # -------------------------------------------------
    # Time integration
    # -------------------------------------------------

    def step(self, velocity, thrust, dt):
        """
        Advance boat velocity one timestep.

        Euler's integration.
        """

        a = self.acceleration(thrust, velocity)

        v_new = velocity + a * dt

        return v_new
