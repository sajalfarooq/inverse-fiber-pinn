import numpy as np
import pytest
from optics_portfolio.fiber import grid,sech,propagate,relative_error
from optics_portfolio.validation import gaussian_exact,sampling_diagnostics,certify_fiber
from optics_portfolio.inverse import field_at,fit_parameters,residual,train_pinn

def test_parameter_recovery_independent_step_resolution():
    t, _ = grid(64, 16)
    initial = np.exp(-t*t/2+0.2j*t*t)
    zs, locations = [0.3, 0.7], np.arange(16,48,2)
    actual = field_at(initial, t, zs, 0.9, 1.2, 600)
    fit = fit_parameters(initial, t, zs, locations, actual[:, locations])
    np.testing.assert_allclose(fit.x, [0.9, 1.2], atol=2e-4)

def test_pinn_residual_on_analytical_soliton():
    import torch
    class Exact(torch.nn.Module):
        log_d = torch.tensor(0., dtype=torch.float64)
        log_g = torch.tensor(0., dtype=torch.float64)
        def forward(self, x):
            envelope = 1/torch.cosh(x[:, 1:2])
            return envelope*torch.cat((torch.cos(x[:, 0:1]/2), torch.sin(x[:, 0:1]/2)), 1)
    x = torch.rand(31, 2, dtype=torch.float64)*2
    assert residual(Exact(), x).abs().max().item() < 1e-14

def test_pinn_plane_wave_residual():
    import torch
    class Exact(torch.nn.Module):
        log_d=torch.tensor(np.log(.9),dtype=torch.float64)
        log_g=torch.tensor(np.log(1.2),dtype=torch.float64)
        def forward(self,x):
            phase=.7*x[:,1:2]+(1.2*.8**2-.9*.7**2/2)*x[:,0:1]
            return .8*torch.cat([phase.cos(),phase.sin()],dim=1)
    x=torch.rand(50,2,dtype=torch.float64)
    assert residual(Exact(),x).abs().max()<1e-14

def test_pinn_training_and_data_only_parameter_guard():
    t,_=grid(32,12)
    x=np.array([(z,tt) for z in [.25,.75] for tt in t[8:24]])
    y=np.array([sech(tt)*np.exp(.5j*z) for z,tt in x])
    model,history=train_pinn(x,y,t,sech(t).astype(complex),steps=60,seed=7,physics_weight=0)
    assert history[-1]["loss"]<history[0]["loss"]
    assert not model.log_d.requires_grad
    assert not model.log_g.requires_grad
    assert all(np.isfinite(h["loss"]) for h in history)

