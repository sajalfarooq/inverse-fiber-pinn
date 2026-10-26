"""Synthetic complex-field parameter inference and a data-augmented PINN."""
import numpy as np
from scipy.optimize import least_squares
from .fiber import propagate


def field_at(u0, t, distances, d, g, steps_per_unit=200):
    return np.array([propagate(u0, t, z, max(1, int(np.ceil(z * steps_per_unit))),
                               d, g, save_every=100000)[1][-1] for z in distances])


def fit_parameters(u0, t, distances, locations, observed, initial=(0.8, 0.8), steps_per_unit=200):
    def residual(p):
        delta = field_at(u0, t, distances, *p, steps_per_unit)[:, locations] - observed
        return np.r_[delta.real.ravel(), delta.imag.ravel()]
    return least_squares(residual, initial, bounds=([0.2, 0.2], [2.0, 2.0]),
                         xtol=1e-9, ftol=1e-9, gtol=1e-9)


def make_pinn(seed=7, width=48):
    import torch
    torch.manual_seed(seed)
    class PINN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(),
                torch.nn.Linear(width, width), torch.nn.Tanh(),
                torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 2))
            self.log_d = torch.nn.Parameter(torch.tensor(0.8).log())
            self.log_g = torch.nn.Parameter(torch.tensor(0.8).log())
        def forward(self, x):
            # Input order: z,t; derivatives in residual include this scaling.
            return self.net(x * x.new_tensor([2.0, 0.3]) - x.new_tensor([1.0, 0.0]))
    return PINN()


def residual(model, coordinates):
    import torch
    x = coordinates.detach().clone().requires_grad_(True)
    out = model(x)
    a, b = out[:, 0:1], out[:, 1:2]
    def gradient(v):
        return torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    da, db = gradient(a), gradient(b)
    att = gradient(da[:, 1:2])[:, 1:2]
    btt = gradient(db[:, 1:2])[:, 1:2]
    d, g = model.log_d.exp(), model.log_g.exp()
    power = a*a + b*b
    # Real: -b_z+d*a_tt/2+g*power*a; imaginary: a_z+d*b_tt/2+g*power*b.
    return torch.cat((-db[:, 0:1] + d*att/2 + g*power*a,
                       da[:, 0:1] + d*btt/2 + g*power*b), dim=1)


def train_pinn(x_data, y_data, t, u0, steps=2500, seed=7, *,
               physics_weight=1.0, data_weight=20.0, lbfgs_steps=0,
               width=48, collocation_count=384, fixed_coefficients=None):
    import torch
    torch.set_num_threads(2)
    if steps < 1 or lbfgs_steps < 0 or physics_weight < 0 or data_weight < 0:
        raise ValueError("Invalid training budget or loss weights")
    model = make_pinn(seed, width)
    if fixed_coefficients is not None:
        with torch.no_grad():
            model.log_d.copy_(torch.tensor(fixed_coefficients[0]).log())
            model.log_g.copy_(torch.tensor(fixed_coefficients[1]).log())
    if fixed_coefficients is not None or physics_weight == 0:
        model.log_d.requires_grad_(False)
        model.log_g.requires_grad_(False)
    tx = lambda a: torch.tensor(a, dtype=torch.float32)
    xd, yd = tx(x_data), tx(np.c_[y_data.real, y_data.imag])
    xi = tx(np.c_[np.zeros(len(t)), t])
    yi = tx(np.c_[u0.real, u0.imag])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.002)
    history = []
    boundary_z = torch.linspace(0, 1, 64)
    left = torch.stack((boundary_z, boundary_z*0+t[0]), 1)
    right = torch.stack((boundary_z, boundary_z*0+t[-1]), 1)
    def sample():
        collocation = torch.rand(collocation_count, 2)
        collocation[:, 1] = collocation[:, 1] * (t[-1]-t[0]) + t[0]
        return collocation
    def terms(collocation):
        physics = residual(model, collocation).square().mean() if physics_weight else torch.tensor(0.)
        data = (model(xd)-yd).square().mean()
        initial = (model(xi)-yi).square().mean()
        boundary = (model(left).square().mean()+model(right).square().mean()) / 2
        loss = physics_weight*physics + data_weight*data + 20*initial + boundary
        return loss, physics, data, initial, boundary
    def log(k, stage, values):
        loss, physics, data, initial, boundary = values
        history.append({"step": k, "stage": stage, "loss": loss.item(), "physics": physics.item(),
                        "data": data.item(), "initial": initial.item(), "boundary": boundary.item(),
                        "d": model.log_d.exp().item(), "g": model.log_g.exp().item()})
    for k in range(steps):
        values = terms(sample())
        loss = values[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if k % 100 == 0 or k == steps-1:
            log(k+1, "Adam", values)
    # Fixed points make each L-BFGS closure a deterministic objective.
    collocation = sample()
    if lbfgs_steps:
        optimizer = torch.optim.LBFGS(parameters, lr=0.5, max_iter=1,
                                     history_size=20, line_search_fn="strong_wolfe")
        for k in range(lbfgs_steps):
            def closure():
                optimizer.zero_grad()
                loss = terms(collocation)[0]
                loss.backward()
                return loss
            optimizer.step(closure)
            if k % 10 == 0 or k == lbfgs_steps-1:
                log(steps+k+1, "LBFGS", terms(collocation))
    return model, history
