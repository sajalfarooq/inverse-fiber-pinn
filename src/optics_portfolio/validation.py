"""Reusable numerical certification and metrics, independent of plotting."""
import numpy as np
from .fiber import grid, sech, propagate, relative_error


def gaussian_exact(t, z, d=1.0):
    """Infinite-domain linear propagation for exp(-t²/2), g=0."""
    q = 1 + 1j*d*np.asarray(z)[..., None]
    return np.exp(-np.asarray(t)**2/(2*q))/np.sqrt(q)


def sampling_diagnostics(field, edge_fraction=0.08):
    """Fraction of energy at temporal edges and upper 20% of FFT frequencies."""
    field = np.asarray(field)
    n = field.shape[-1]
    k = max(1, int(edge_fraction*n))
    p = abs(field)**2
    norm = p.sum(axis=-1)
    spectrum = abs(np.fft.fft(field, axis=-1))**2
    freq = abs(np.fft.fftfreq(n))
    return {"edge_energy_fraction": float(np.max((p[..., :k].sum(-1)+p[..., -k:].sum(-1))/norm)),
            "high_frequency_energy_fraction": float(np.max(spectrum[..., freq >= 0.4].sum(-1)/spectrum.sum(-1)))}


def certify_fiber():
    """Thresholds target a resolved reference case, not every possible pulse."""
    t, _ = grid(1024, 40)
    initial = sech(t)
    z, field = propagate(initial, t, 2, 800, save_every=40)
    exact = np.exp(0.5j*z[:, None])*initial
    errors = []
    for steps in [40, 80, 160]:
        actual = propagate(initial, t, 1, steps, save_every=steps)[1][-1]
        errors.append(relative_error(actual, initial*np.exp(0.5j)))
    order = float(np.polyfit(np.log(1/np.array([40, 80, 160])), np.log(errors), 1)[0])
    linear = propagate(np.exp(-t*t/2), t, 2, 40, g=0, save_every=40)[1][-1]
    gaussian_error = relative_error(linear, gaussian_exact(t, 2))
    soliton_error = relative_error(field, exact)
    checks = [
        {"name": "fundamental_soliton_complex_field", "value": soliton_error, "limit": 2e-5,
         "passed": soliton_error < 2e-5},
        {"name": "gaussian_infinite_domain_solution", "value": gaussian_error, "limit": 1e-10,
         "passed": gaussian_error < 1e-10},
        {"name": "strang_convergence_order", "value": order, "expected_interval": [1.8, 2.2],
         "passed": 1.8 < order < 2.2},
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "sampling": sampling_diagnostics(field), "step_errors": errors}


def aggregate(values):
    values = np.asarray(values, float)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()), "maximum": float(values.max())}
