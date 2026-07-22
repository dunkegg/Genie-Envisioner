"""Recovered Za-adaptor server command-line entrypoint."""
from _recovered_loader import load_recovered, run_recovered
if __name__ == "__main__":
    run_recovered(__file__, "_main_za_adaptor_server_recovered")
else:
    load_recovered(globals(), __file__, "_main_za_adaptor_server_recovered")

