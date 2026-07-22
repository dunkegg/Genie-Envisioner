"""Recovered Za checkpoint metrics tool."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_inspect_za_pt_metrics_recovered")
else:
    load_recovered(globals(), __file__, "_inspect_za_pt_metrics_recovered")

