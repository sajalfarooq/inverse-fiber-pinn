import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from optics_portfolio.fiber import grid, relative_error
from optics_portfolio.inverse import field_at, fit_parameters, train_pinn
from optics_portfolio.output import setup, write_metrics, plt


def main(steps=2500):
    out = setup(Path(__file__).parent)
    rng = np.random.default_rng(7)
    t, _ = grid(128, 16)
    # A nonstationary chirped Gaussian avoids using only a shape-preserving soliton.
    u0 = np.exp(-t*t/2 + 0.25j*t*t)
    distances = np.array([0.25, 0.5, 0.75, 1.0])
    truth = np.array([0.9, 1.2])
    # Generate with a finer step than fitting to reduce inverse-crime bias.
    clean = field_at(u0, t, distances, *truth, steps_per_unit=800)
    locations = np.arange(32, 96, 2)
    noise_levels = [0.0, 0.01, 0.03, 0.05]
    rows = []
    start = perf_counter()
    for noise in noise_levels:
        estimates = []
        for _ in range(3):
            noisy = clean[:, locations] + noise/np.sqrt(2)*(rng.normal(size=(4, len(locations)))
                                                         + 1j*rng.normal(size=(4, len(locations))))
            fit = fit_parameters(u0, t, distances, locations, noisy)
            estimates.append(fit.x)
        values = np.array(estimates)
        rows.append({"noise_std_absolute_field": noise, "estimates": values.tolist(),
                     "mean": values.mean(0).tolist(), "std": values.std(0).tolist()})
    classical_time = perf_counter()-start
    observed = clean[:, locations] + 0.01/np.sqrt(2)*(rng.normal(size=(4, len(locations)))
                                                   + 1j*rng.normal(size=(4, len(locations))))
    start = perf_counter()
    fit = fit_parameters(u0, t, distances, locations, observed)
    fit_time = perf_counter()-start
    # Normalized Jacobian singular values diagnose local parameter ambiguity.
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    x = np.array([(z, tt) for z in distances for tt in t[locations]])
    start = perf_counter()
    model, history = train_pinn(x, observed.ravel(), t, u0, steps=steps)
    train_time = perf_counter()-start
    # Dense evaluation includes measured and unmeasured points; report a disjoint subset separately.
    xv = np.array([(z, tt) for z in distances for tt in t])
    start = perf_counter()
    with torch.no_grad():
        outputs = model(torch.tensor(xv, dtype=torch.float32)).numpy()
    inference_time = perf_counter()-start
    prediction = (outputs[:, 0]+1j*outputs[:, 1]).reshape(clean.shape)
    held_out = np.setdiff1d(np.arange(128), locations)
    recovered = [model.log_d.exp().item(), model.log_g.exp().item()]
    torch.save({"state_dict": model.state_dict(), "seed": 7, "steps": steps, "width": 48}, out / "pinn.pt")
    np.savez_compressed(out / "study_data.npz", t=t, z=distances, clean=clean,
                        observed=observed, locations=locations, prediction=prediction)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for ax, image, title in zip(axes, [abs(clean)**2, abs(prediction)**2, abs(prediction-clean)],
                               ["SSFM reference intensity", "PINN intensity", "Complex-field absolute error"]):
        im = ax.pcolormesh(t, distances, image, shading="auto")
        ax.set(xlabel="Retarded time tau", ylabel="Distance xi", title=title)
        fig.colorbar(im, ax=ax)
    fig.savefig(out / "pinn_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for i, label in enumerate(["Dispersion d", "Nonlinearity g"]):
        axes[0].errorbar(noise_levels, [r["mean"][i] for r in rows],
                         yerr=[r["std"][i] for r in rows], fmt="o-", label=label)
        axes[0].axhline(truth[i], linestyle="--", alpha=0.5)
    axes[0].set(xlabel="Absolute complex noise std", ylabel="Recovered coefficient", title="Classical fit: 3 noise draws")
    axes[0].legend()
    axes[1].semilogy([h["step"] for h in history], [h["loss"] for h in history])
    axes[1].set(xlabel="Adam step", ylabel="Weighted loss", title="Data-augmented inverse PINN")
    fig.savefig(out / "inference.png")
    plt.close(fig)
    write_metrics(out, {"seed": 7, "true_parameters_d_g": truth.tolist(), "classical_noise_study": rows,
        "classical_noise_study_seconds": classical_time,
        "classical_shared_data_parameters": fit.x.tolist(), "classical_shared_data_seconds": fit_time,
        "jacobian_condition_number": float(singular[0]/singular[-1]),
        "pinn_parameters": recovered, "pinn_steps": steps,
        "pinn_training_seconds": train_time, "pinn_dense_inference_seconds": inference_time,
        "pinn_held_out_relative_field_error": relative_error(prediction[:, held_out], clean[:, held_out]),
        "pinn_relative_parameter_errors": (abs(np.array(recovered)-truth)/truth).tolist(),
        "pinn_history": history, "measurement_type": "phase-resolved complex field, not intensity only",
        "noise_definition": "E[abs(noise)^2]=sigma^2; input peak amplitude=1",
        "training_data_points": len(x), "held_out_points": len(held_out)*len(distances)})
    print("Inverse study saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    main(args.steps)
