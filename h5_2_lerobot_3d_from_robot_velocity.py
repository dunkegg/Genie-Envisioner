"""Recovered 3-D robot-velocity conversion tool."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_h5_2_lerobot_3d_from_robot_velocity_recovered")
else:
    load_recovered(globals(), __file__, "_h5_2_lerobot_3d_from_robot_velocity_recovered")

