"""Periodic spectral solver for i*u_z + d/2*u_tt + g*|u|^2*u = 0.

z and t are dimensionless. Real d and g imply unitary substeps. The FFT
domain is periodic; energy conservation alone does not certify accuracy.
"""
import numpy as np


def grid(n=512, window=24.0):
    if not isinstance(n, (int, np.integer)) or n < 8 or not np.isfinite(window) or window <= 0:
        raise ValueError("Use n >= 8 and a positive window")
    t = np.linspace(-window / 2, window / 2, n, endpoint=False)
    return t, 2 * np.pi * np.fft.fftfreq(n, d=window / n)


def sech(t):
    return 1 / np.cosh(t)


def propagate(u0, t, zmax=2.0, steps=400, d=1.0, g=1.0, save_every=1):
    t = np.asarray(t, float)
    u = np.array(u0, dtype=complex, copy=True)
    if t.ndim != 1 or len(t) < 8 or u.shape != t.shape:
        raise ValueError("u0 and t must be matching 1D arrays of length >= 8")
    if not isinstance(steps, (int, np.integer)) or not isinstance(save_every, (int, np.integer)):
        raise ValueError("Step count and save interval must be integers")
    if steps < 1 or save_every < 1 or not np.isfinite(zmax) or zmax < 0:
        raise ValueError("Positive distance, step count and save interval required")
    if not all(np.all(np.isfinite(v)) for v in (t, u, d, g)) or np.iscomplexobj(d) or np.iscomplexobj(g):
        raise ValueError("Finite fields and real finite coefficients are required")
    dt = t[1] - t[0]
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-13):
        raise ValueError("t must be uniformly increasing")
    w = 2 * np.pi * np.fft.fftfreq(len(t), dt)
    h = zmax / steps
    half = np.exp(-0.25j * d * w**2 * h)
    hist, zs = [u.copy()], [0.0]
    for k in range(1, steps + 1):
        u = np.fft.ifft(np.fft.fft(u) * half)
        u *= np.exp(1j * g * np.abs(u)**2 * h)
        u = np.fft.ifft(np.fft.fft(u) * half)
        if k % save_every == 0 or k == steps:
            hist.append(u.copy())
            zs.append(k * h)
    return np.array(zs), np.array(hist)


def diagnostics(u, t, d=1.0, g=1.0):
    dt = t[1] - t[0]
    p = np.abs(u)**2
    energy = p.sum(axis=-1) * dt
    mean = (p * t).sum(axis=-1) * dt / energy
    width = np.sqrt((p * (t - np.expand_dims(mean, -1))**2).sum(axis=-1) * dt / energy)
    w = 2 * np.pi * np.fft.fftfreq(len(t), dt)
    ut = np.fft.ifft(1j * w * np.fft.fft(u, axis=-1), axis=-1)
    ham = (d / 2 * np.abs(ut)**2 - g / 2 * p**2).sum(axis=-1) * dt
    return {"energy": energy, "rms_width": width, "hamiltonian": ham}


def relative_error(prediction, reference):
    if np.shape(prediction) != np.shape(reference) or np.linalg.norm(reference) == 0:
        raise ValueError("Matching arrays and a nonzero reference are required")
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))
