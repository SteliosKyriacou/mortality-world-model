import numpy as np
import torch

from mwm.data.synthetic import SynthConfig, simulate_cohort
from mwm.encoders.hetgnn import HetGNNEncoder
from mwm.heads.heads import FeatureDecoder


def _data():
    long, out, truth = simulate_cohort(SynthConfig(n_persons=600, seed=3, n_genes=100), None, verbose=False)
    from mwm.data.common import long_to_visits
    long = long[long.feature != "u_nutrient"]
    units = long.drop_duplicates("feature").set_index("feature")["unit"]
    clin = sorted(units.index[units != "NPX"])
    om = sorted(units.index[units == "NPX"])
    vt = long_to_visits(long, clin + om)
    X = vt.X.copy()
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-6)
    # pathways = consecutive blocks of 25 genes (as generated)
    pw = [list(range(i, min(i + 25, len(om)))) for i in range(0, len(om), 25)]
    return X.astype(np.float32), len(clin), len(om), pw


def test_hetgnn_missing_modalities_and_training():
    torch.manual_seed(0)
    X, nc, no, pw = _data()
    M = ~np.isnan(X)
    has_om = M[:, nc:].any(1)
    assert has_om.any() and (~has_om).any()
    enc = HetGNNEncoder(nc, no, pw, d_latent=8, hidden=32)
    dec = FeatureDecoder(8, X.shape[1], hidden=64)
    x, m = torch.tensor(np.nan_to_num(X)), torch.tensor(M)
    z = enc(x[:64], m[:64])
    assert z.shape == (64, 8) and torch.isfinite(z).all()
    # a visit without omics: changing (unobserved) omics values must not change z
    i = int(np.nonzero(~has_om)[0][0])
    x2 = x[i:i + 1].clone()
    x2[:, nc:] = 5.0
    assert torch.allclose(enc(x[i:i + 1], m[i:i + 1]), enc(x2, m[i:i + 1]), atol=1e-6)
    # a visit with omics: dropping the omics modality changes z but still works
    j = int(np.nonzero(has_om)[0][0])
    mj = m[j:j + 1].clone()
    mj[:, nc:] = False
    assert not torch.allclose(enc(x[j:j + 1], m[j:j + 1]), enc(x[j:j + 1], mj))
    # short masked-autoencoder training lowers reconstruction loss; pathway embeddings get grads
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 3e-3)
    losses = []
    for step in range(60):
        b = torch.randint(0, len(x), (128,))
        hide = (torch.rand(128, X.shape[1]) < 0.25) & m[b]
        loss = dec.nll(enc(x[b], m[b] & ~hide), x[b], m[b].float())
        opt.zero_grad()
        loss.backward()
        if step == 0:
            assert enc.path_e.grad is not None and enc.path_e.grad.abs().sum() > 0
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
