
"""Verify the retained release without training or modifying any files."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((ROOT/"published/manifest.json").read_text())
    for name,expected in manifest["files"].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT): raise ValueError("Invalid manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("Retained artifact changed: "+name)
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    import torch
    from optics_portfolio.inverse import make_pinn
    from optics_portfolio.fiber import relative_error
    torch.set_num_threads(2)
    metrics=json.loads((ROOT/"results/advanced_metrics.json").read_text())
    with np.load(ROOT/"results/advanced_data.npz") as a:
        t,z,truth=a["t"],a["z_test"],a["unseen_reference"]
        coordinates=torch.tensor([(zz,tt) for zz in z for tt in t],dtype=torch.float32)
        for row in metrics["runs"]:
            state=torch.load(ROOT/f"results/{row['mode']}_seed{row['seed']}.pt",map_location="cpu",weights_only=True)
            model=make_pinn(row["seed"],state["width"])
            model.load_state_dict(state["state_dict"])
            with torch.no_grad(): output=model(coordinates).numpy()
            field=(output[:,0]+1j*output[:,1]).reshape(truth.shape)
            error=relative_error(field[:,abs(t)<4],truth[:,abs(t)<4])
            np.testing.assert_allclose(error,row["unseen_distance_pulse_relative_field_error"],rtol=2e-5,atol=2e-7)
    return len(manifest["files"])

if __name__=="__main__": print("Verified",verify(),"retained artifact files; no retraining")
