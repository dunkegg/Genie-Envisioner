"""Recovered TD-MPC latent validation and repair tool."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_validate_repair_tdmpc_latents_recovered")
else:
    load_recovered(globals(), __file__, "_validate_repair_tdmpc_latents_recovered")

