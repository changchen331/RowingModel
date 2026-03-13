import matplotlib.pyplot as plt

from simulator import RowingSimulator

# rowing_simulation/
# │
# ├── geometry.py        # 几何关系
# ├── kinematics.py      # 位移 / 角度 / 速度关系
# ├── dynamics.py        # 力学模型
# ├── hydrodynamics.py   # 水动力阻力
# ├── rower_model.py     # 划手施力模型
# ├── boat_model.py      # 船体动力学
# ├── simulator.py       # ODE求解主循环
# │
# └── main.py            # 运行入口

if __name__ == '__main__':
    sim = RowingSimulator()

    result = sim.run(20)

    t = result.t
    v = result.y[0]

    plt.plot(t, v)
    plt.xlabel("time (s)")
    plt.ylabel("boat velocity (m/s)")
    plt.title("Boat velocity vs time")

    plt.show()
