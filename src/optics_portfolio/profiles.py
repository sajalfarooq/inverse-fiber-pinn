"""Explicit experiment budgets. Smoke outputs are not accuracy claims."""
PROFILES = {
    "smoke": {"fiber_steps": 300, "seeds": [7], "pinn_steps": 100,
              "lbfgs_steps": 0, "mc_trials": 8, "optimizer_iterations": 5, "phase_iterations": 30},
    "standard": {"fiber_steps": 2400, "seeds": [7, 19, 31], "pinn_steps": 3000,
                 "lbfgs_steps": 40, "mc_trials": 150, "optimizer_iterations": 65, "phase_iterations": 250},
    "research": {"fiber_steps": 4800, "seeds": [7, 19, 31, 43, 59], "pinn_steps": 6000,
                 "lbfgs_steps": 100, "mc_trials": 500, "optimizer_iterations": 150, "phase_iterations": 500},
}


def profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}")
    return dict(PROFILES[name])
