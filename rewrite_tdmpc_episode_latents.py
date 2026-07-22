"""Recovered per-episode latent rewrite tool (Python 3.13 cache)."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_rewrite_tdmpc_episode_latents_recovered")
else:
    load_recovered(globals(), __file__, "_rewrite_tdmpc_episode_latents_recovered")

