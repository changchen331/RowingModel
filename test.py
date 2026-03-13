from boat_model import BoatModel
from geometry import OarGeometry
from hydrodynamics import BladeHydrodynamics
from kinematics import OarKinematics

geo = OarGeometry(0.9, 2.1)

kin = OarKinematics(geo)

hydro = BladeHydrodynamics(geo)

t = 0.3
boat_velocity = 3.5

v_blade = kin.blade_velocity(t)

F_blade = hydro.blade_force(v_blade, boat_velocity)

thrust = hydro.boat_thrust(F_blade)

print("blade velocity:", v_blade)
print("blade force:", F_blade)
print("boat thrust:", thrust)
print()

# Boat model
boat = BoatModel()

velocity = 3.5
thrust = 2000
dt = 0.01

for i in range(10):
    velocity = boat.step(velocity, thrust, dt)
    print(velocity)
