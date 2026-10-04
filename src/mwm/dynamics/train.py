"""Training loops for the dynamics models (same optimiser/budget for A and B)."""
from __future__ import annotations

import copy
import gc
import time

import numpy as np
import torch

from .flow import FlowModel, SnapshotOTFlow, StochasticFlow
from .sde import NeuralSDE


def pairs_to_tensors(p: dict, device="cuda"):
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return (t(p["z0"]), t(p["z1"]), t(p["a0"]), t(p["a1"]), t(p["u"]).reshape(len(p["a0"]), -1))


def _cost_start(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return time.time()


def _cost_end(t0, device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        mem = float("nan")
    return {"train_seconds": time.time() - t0, "peak_vram_mb": mem}


def train_pairs(model, train_p: dict, val_p: dict, epochs=200, batch=512, lr=2e-3, device="cuda",
                patience=30, log=None, loss_kw=None, val_every=5, seed=0):
    """Generic loop: works for NeuralSDE and FlowModel (both expose .loss/.val_loss)."""
    torch.manual_seed(seed)
    loss_kw = loss_kw or {}
    model.to(device)
    tr = pairs_to_tensors(train_p, device)
    va = pairs_to_tensors(val_p, device)
    stage1 = isinstance(model, StochasticFlow)  # B': velocity first, diffusion in stage 2
    params = [p for n, p in model.named_parameters()
              if not (stage1 and (n.startswith(("diff.", "score.")) or n == "log_r"))]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = tr[0].shape[0]
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = _cost_start(device)
    model.nfe = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            loss, _ = model.loss(tuple(x[b] for x in tr), **loss_kw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(b)
            if getattr(model, "solver", "native") != "native" and (s // batch) % 8 == 0:
                gc.collect()  # torchsde objects form reference cycles holding graph memory
        sched.step()
        if ep % val_every == 0 or ep == epochs - 1:
            model.eval()
            vl = model.val_loss(va)
            hist.append({"epoch": ep, "train": tot / n, "val": vl})
            if log:
                log(f"[{type(model).__name__}] ep {ep} train {tot/n:.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += val_every
                if bad >= patience:
                    break
    cost = _cost_end(t0, device)
    cost["train_nfe"] = model.nfe
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"history": hist, "best_val": best, **cost}


def fit_bprime(model: StochasticFlow, train_p, val_p, states_z, states_a, states_u,
               dsm_iters=4000, diff_epochs=40, batch=512, lr=2e-3, K=32, n_steps=32,
               device="cuda", log=None, seed=0):
    """Stages 2-3 of B' (velocity already trained by `train_pairs`):
    (2) denoising score matching on train visit states; (3) diffusion + obs noise by
    simulated-pair NLL with v and s frozen."""
    torch.manual_seed(seed)
    model.to(device).train()
    t0 = _cost_start(device)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    Zs, As, Us = t(states_z), t(states_a), t(states_u).reshape(len(states_a), -1)
    opt = torch.optim.AdamW(model.score.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, dsm_iters)
    for it in range(dsm_iters):
        b = torch.randint(0, len(Zs), (batch,), device=device)
        l = model.score.dsm_loss(Zs[b], As[b], Us[b])
        opt.zero_grad()
        l.backward()
        opt.step()
        sched.step()
        if log and it % 2000 == 0:
            log(f"[B' DSM] it {it} loss {l.item():.4f}")
    for p in list(model.field.parameters()) + list(model.score.parameters()):
        p.requires_grad_(False)
    params = list(model.diff.parameters()) + ([model.log_r] if model.log_r.requires_grad else [])
    opt = torch.optim.Adam(params, lr=lr)
    tr, va = pairs_to_tensors(train_p, device), pairs_to_tensors(val_p, device)
    n = tr[0].shape[0]
    best, best_state = np.inf, None
    model.nfe = 0
    for ep in range(diff_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            l = model.diffusion_loss(tuple(x[b] for x in tr), K, n_steps)
            if not torch.isfinite(l):
                continue
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        with torch.no_grad():  # keep train-mode start noise so val NLL is comparable
            vl = model.diffusion_loss(va, K, n_steps).item()
        if np.isfinite(vl) and vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
        if log and ep % 10 == 0:
            log(f"[B' diffusion] ep {ep} val nll {vl:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    for n, p in model.named_parameters():
        p.requires_grad_(not (n == "log_r" and getattr(model, "obs_r2_fixed", None) is not None))
    model.eval()
    return {"best_val_nll": best, **_cost_end(t0, device), "train_nfe": model.nfe}


@torch.no_grad()
def _ode(model, z0, a0, a1, u, n_steps):
    sto = model.stochastic
    model.stochastic = False
    try:
        return model.sample(z0, a0, a1, u, n_samples=1, n_steps=n_steps)[0]
    finally:
        model.stochastic = sto


def train_snapshot(model: SnapshotOTFlow, z: np.ndarray, age: np.ndarray, u: np.ndarray,
                   bin_width=5.0, iters=3000, batch=512, lr=2e-3, device="cuda", seed=0, log=None):
    """OT-CFM on cross-sectional (one visit per person) latents between adjacent age bins."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    edges = np.arange(np.floor(age.min()), np.ceil(age.max()) + bin_width, bin_width)
    bins = [np.nonzero((age >= lo) & (age < lo + bin_width))[0] for lo in edges[:-1]]
    pairs = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)
             if len(bins[i]) > 50 and len(bins[i + 1]) > 50]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    t0 = _cost_start(device)
    for it in range(iters):
        A, Bb = pairs[rng.integers(len(pairs))]
        ia = rng.choice(A, min(batch, len(A)), replace=False)
        ib = rng.choice(Bb, min(batch, len(Bb)), replace=False)
        pa, pb = SnapshotOTFlow.ot_pairs(z[ia], z[ib], rng)
        i0, i1 = ia[pa], ib[pb]
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        a0, a1 = t(age[i0]), t(age[i1])
        ok = (a1 - a0) > 0.5
        loss = model.cfm_loss(t(z[i0])[ok], t(z[i1])[ok], a0[ok], a1[ok], t(u[i0]).reshape(len(i0), -1)[ok])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and it % 1000 == 0:
            log(f"[OT-CFM] it {it} loss {loss.item():.4f}")
    model.eval()
    return _cost_end(t0, device)
