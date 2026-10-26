"""Multi-seed inverse PINN ablations with unseen-distance validation."""
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from optics_portfolio.fiber import grid, relative_error
from optics_portfolio.inverse import field_at, fit_parameters, train_pinn, make_pinn, residual
from optics_portfolio.profiles import profile
from optics_portfolio.validation import aggregate
from optics_portfolio.output import result_dir, save_json, plt


def evaluate(model, t, distances):
    x = np.array([(z, tt) for z in distances for tt in t], dtype=np.float32)
    with torch.no_grad():
        values = model(torch.tensor(x)).numpy()
    return (values[:,0]+1j*values[:,1]).reshape(len(distances),len(t))


def main(name="standard"):
    cfg = profile(name)
    out = result_dir(Path(__file__).parent)
    out.mkdir(parents=True,exist_ok=True)
    t,_ = grid(128,16)
    u0 = np.exp(-t*t/2+0.25j*t*t)
    zs = np.array([0.25,0.5,0.75,1.0])
    test_z = np.array([0.125,0.375,0.625,0.875])
    truth = np.array([0.9,1.2])
    clean = field_at(u0,t,zs,*truth,steps_per_unit=1000)
    unseen = field_at(u0,t,test_z,*truth,steps_per_unit=1000)
    locations = np.arange(32,96,2)
    rng = np.random.default_rng(2026)
    observed = clean[:,locations]+0.01/np.sqrt(2)*(rng.normal(size=(4,32))+1j*rng.normal(size=(4,32)))
    x = np.array([(z,tt) for z in zs for tt in t[locations]])
    start = perf_counter()
    fit = fit_parameters(u0,t,zs,locations,observed)
    classical_seconds = perf_counter()-start
    fitted_field = field_at(u0,t,test_z,*fit.x,steps_per_unit=400)
    pulse = abs(t)<4
    rows=[]
    prediction_first=None
    # Physics-only with known coefficients is a forward ablation; with unknown
    # coefficients and no observations this inverse problem is underdetermined.
    modes = [("joint_inverse",1.,20.,None), ("data_only",0.,20.,None),
             ("physics_only_known_coefficients",1.,0.,truth)]
    for mode,pw,dw,fixed in modes:
        seeds=cfg["seeds"] if mode!="physics_only_known_coefficients" else cfg["seeds"][:1]
        for seed in seeds:
            print(f"Training {mode}, seed={seed}, Adam={cfg['pinn_steps']}",flush=True)
            start=perf_counter()
            model,history=train_pinn(x,observed.ravel(),t,u0,steps=cfg["pinn_steps"],seed=seed,
                physics_weight=pw,data_weight=dw,lbfgs_steps=cfg["lbfgs_steps"],fixed_coefficients=fixed)
            seconds=perf_counter()-start
            pred=evaluate(model,t,test_z)
            d,g=model.log_d.exp().item(),model.log_g.exp().item()
            probe=torch.tensor(np.c_[np.linspace(0.05,0.95,64),np.linspace(-4,4,64)],dtype=torch.float32)
            pde=float(residual(model,probe).square().mean().detach())
            row={"mode":mode,"seed":seed,"train_seconds":seconds,
                 "unseen_distance_relative_field_error":relative_error(pred,unseen),
                 "unseen_distance_pulse_relative_field_error":relative_error(pred[:,pulse],unseen[:,pulse]),
                 "probe_residual_mse":pde,"coefficients_are_inferred":mode=="joint_inverse",
                 "parameters_d_g":[d,g],"history":history}
            if mode=="joint_inverse":
                row["relative_parameter_errors"]=(abs(np.array([d,g])-truth)/truth).tolist()
            rows.append(row)
            torch.save({"state_dict":model.state_dict(),"width":48,"seed":seed,"profile":name,
                        "mode":mode,"true_coefficients":truth.tolist(),"adam_steps":cfg["pinn_steps"],
                        "lbfgs_steps":cfg["lbfgs_steps"]},out/f"{mode}_seed{seed}.pt")
            if mode=="joint_inverse" and seed==cfg["seeds"][0]:
                prediction_first=pred
    summary={mode:aggregate([r["unseen_distance_pulse_relative_field_error"] for r in rows if r["mode"]==mode])
             for mode,_,_,_ in modes}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    vmax=float(np.max(abs(unseen)**2))
    for ax,array,title in zip(axes,[abs(unseen)**2,abs(prediction_first)**2,
                                   np.log10(np.maximum(abs(prediction_first-unseen),1e-8))],
                              ["Unseen-distance SSFM","Joint inverse PINN: first seed","log10 absolute field error"]):
        kwargs={"vmin":0,"vmax":vmax} if "error" not in title else {}
        im=ax.pcolormesh(t,test_z,array,shading="auto",**kwargs)
        ax.set(xlim=(-5,5),xlabel="Retarded time",ylabel="Unseen distance xi",title=title)
        fig.colorbar(im,ax=ax)
    fig.savefig(out/"unseen_distance_comparison.png"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for i,(mode,_,_,_) in enumerate(modes):
        vals=[r["unseen_distance_pulse_relative_field_error"] for r in rows if r["mode"]==mode]
        axes[0].scatter([i]*len(vals),vals)
    axes[0].axhline(relative_error(fitted_field[:,pulse],unseen[:,pulse]),color="black",ls="--",label="Classical fit")
    axes[0].set(xticks=[0,1,2],xticklabels=["Joint inverse","Data only","Physics only\nknown d,g"],
                ylabel="Pulse-region relative field error",yscale="log",title="Ablation at unseen distances")
    axes[0].legend()
    for row in rows:
        if row["mode"]=="joint_inverse":
            h=row["history"]
            axes[1].semilogy([r["step"] for r in h],[r["loss"] for r in h],label=f"seed {row['seed']}")
    axes[1].set(xlabel="Optimizer outer step",ylabel="Weighted loss",title="Independent training seeds")
    axes[1].legend()
    fig.savefig(out/"ablation.png"); plt.close(fig)
    # Same-grid forward timing: repeated medians; training is excluded and reported separately.
    timings={"ssfm":[],"pinn":[]}
    first=make_pinn()
    checkpoint=torch.load(out/f"joint_inverse_seed{cfg['seeds'][0]}.pt",weights_only=True)
    first.load_state_dict(checkpoint["state_dict"])
    evaluate(first,t,test_z)  # warm up
    for _ in range(5):
        start=perf_counter(); field_at(u0,t,test_z,*truth,steps_per_unit=400)
        timings["ssfm"].append(perf_counter()-start)
        start=perf_counter(); evaluate(first,t,test_z)
        timings["pinn"].append(perf_counter()-start)
    np.savez_compressed(out/"advanced_data.npz",t=t,z_train=zs,z_test=test_z,u0=u0,
                        observed=observed,locations=locations,unseen_reference=unseen,
                        first_seed_prediction=prediction_first,classical_prediction=fitted_field)
    save_json(out/"advanced_metrics.json",{"profile":name,"config":cfg,"true_parameters":truth.tolist(),
        "measurement_noise_std":0.01,"measurement_seed":2026,"training_points":len(x),
        "validation":"four unseen distances inside training interval; no labels at these distances",
        "pulse_mask":"abs(tau)<4","classical_parameters":fit.x.tolist(),
        "classical_fit_seconds":classical_seconds,
        "classical_unseen_pulse_error":relative_error(fitted_field[:,pulse],unseen[:,pulse]),
        "ablation_summary":summary,"runs":rows,"forward_timing_seconds":timings,
        "forward_timing_median_seconds":{k:float(np.median(v)) for k,v in timings.items()},
        "timing_scope":"same four distances and 128 time samples; SSFM steps/unit=400; PINN includes tensor conversion",
        "note":"Data-only coefficients are frozen, not inferred. Physics-only uses true known coefficients."})
    print("Extended inverse study complete",flush=True)


if __name__=="__main__":
    import os
    main(os.environ.get("OPTICS_PROFILE","standard"))
