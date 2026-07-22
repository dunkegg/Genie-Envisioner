"""Recovered TD-MPC H5 to LeRobot conversion tool."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_h5_tdmpc_2_lerobot_recovered")
else:
    load_recovered(globals(), __file__, "_h5_tdmpc_2_lerobot_recovered")

