from scipy.integrate import solve_ivp

from boat_model import BoatModel
from geometry import OarGeometry
from hydrodynamics import BladeHydrodynamics
from kinematics import OarKinematics
from rower_model import RowerModel


class RowingSimulator:

    def __init__(self):
        # -------------------------------------------------
        # Create models
        # -------------------------------------------------

        self.geometry = OarGeometry(
            l_in=0.9,
            l_out=2.1
        )

        self.rower = RowerModel()

        self.kinematics = OarKinematics(
            geometry=self.geometry,
            rower_model=self.rower
        )

        self.hydro = BladeHydrodynamics(
            geometry=self.geometry
        )

        self.boat = BoatModel(
            mass=90,
            drag_coeff=30
        )

    # -------------------------------------------------
    # System dynamics
    # -------------------------------------------------

    def dynamics(self, t, state):
        """
        ODE system.

        state = [boat_velocity]
        """

        boat_velocity = state[0]

        # 1. blade velocity
        blade_v = self.kinematics.blade_velocity(t)

        # 2. hydrodynamic force
        blade_force = self.hydro.blade_force(
            blade_v,
            boat_velocity
        )

        # 3. thrust on boat
        thrust = self.hydro.boat_thrust(blade_force)

        # 4. boat acceleration
        dvdt = self.boat.acceleration(
            thrust,
            boat_velocity
        )

        return [dvdt]

    # -------------------------------------------------
    # Run simulation
    # -------------------------------------------------

    def run(self, t_final=20):
        initial_state = [3.0]

        solution = solve_ivp(
            self.dynamics,
            [0, t_final],
            initial_state,
            max_step=0.01
        )

        return solution
