"""Recovered TD-MPC2 external-Z test tool."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_isaac_test_tdmpc2_external_z_recovered")
else:
    load_recovered(globals(), __file__, "_isaac_test_tdmpc2_external_z_recovered")

