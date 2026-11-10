"""Deterministic linear inference (DLI, the Backus-Gilbert-Parker bounds) on pygeoinf0.

A port of ../convex_analysis/dli.ipynb (intervalinf + pygeoinf 1.x, kept unchanged for reference): the same problem
on L2([0, 1]) with 50 seeded "normal-mode" data kernels, 20 unit-mass C-infinity bump targets of support width 0.2, the
same true model, noise draw, model ball (centre m0(x) = x, radius 1.05 ||m_bar - m0||) and data ball (radius 1.05
||noise||). The notebook bounded each property by 40 KKT solves (fsolve); here one data-space decomposition gives every
bound in closed form with a certificate (EXACT when primal and dual agree, else a stated bracket).

    python dli_pygeoinf0.py            # prints the bounds; --plot writes dli_pygeoinf0.png
"""
import sys
import time

import numpy as np

import pygeoinf0 as pg

N_D, N_P, WIDTH = 50, 20, 0.2
RESOLUTION = 0.002                       # narrowest feature: Gaussian envelopes of width >= 0.01, periods >= 0.05


def normal_mode(index, base=2, n_modes_range=(1, 50), coeff_range=(-5, 5), freq_range=(0.1, 20),
                width_percent_range=(1, 5)):
    """intervalinf's NormalModesProvider.get_function_by_index on [0, 1]: the same draws from the same seed."""
    rng = np.random.RandomState((base + index * 1000003) % (2**31 - 1))
    n = rng.randint(n_modes_range[0], n_modes_range[1] + 1)
    c = rng.uniform(coeff_range[0], coeff_range[1], n)
    f = rng.uniform(freq_range[0], freq_range[1], n)
    centre = rng.uniform(0.0, 1.0)
    width = rng.uniform(*width_percent_range) / 100.0

    def kernel(x):
        x = np.asarray(x, float)
        return np.sin(2 * np.pi * np.multiply.outer(x, f)) @ c * np.exp(-0.5 * ((x - centre) / width) ** 2)
    return kernel


def target(c, width=WIDTH, k=1.0):
    """intervalinf's BumpFunctionProvider kernel: exp(k t^2/(t^2 - 1)), t = 2 (x - c)/width, of unit mass. (The bank's
    bump is the same shape, but it refuses a support that touches the boundary, as the two end targets here do.)"""
    from scipy import integrate
    Z = integrate.quad(lambda t: np.exp(k * t * t / (t * t - 1)), -1, 1, epsabs=1e-14, epsrel=1e-13, limit=200)[0] * width / 2

    def f(x):
        t = 2.0 * (np.asarray(x, float) - c) / width
        inside = np.abs(t) < 1
        out = np.zeros_like(t)
        out[inside] = np.exp(k * t[inside] ** 2 / (t[inside] ** 2 - 1))
        return out / Z
    return f


def build():
    I = pg.Interval("[0, 1]")
    M = pg.L2(I)
    D, P = pg.Euclidean(N_D), pg.Euclidean(N_P)
    G = pg.Functionals([M(normal_mode(i), resolution=RESOLUTION) for i in range(N_D)], into=D, name="G")
    centres = np.linspace(WIDTH / 2, 1 - WIDTH / 2, N_P)
    T = pg.Functionals([M(target(c), breaks=[b for b in (c - WIDTH / 2, c + WIDTH / 2) if 0 < b < 1], resolution=WIDTH / 20)
                        for c in centres], into=P, name="T")
    m_bar = M(lambda x: np.exp(-((x - 0.5) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x, resolution=0.05)
    m0 = M(lambda x: np.asarray(x, float), resolution=1.0)
    d_bar = G(m_bar).values
    np.random.seed(42)                                   # the notebook's noise draw
    noise = np.random.normal(0, 0.1 * np.max(d_bar), d_bar.shape)
    d = D(d_bar + noise)
    return dict(M=M, D=D, P=P, G=G, T=T, m_bar=m_bar, m0=m0, d=d, d_bar=d_bar, centres=centres,
                data_radius=1.05 * np.linalg.norm(noise), model_radius=1.05 * (m_bar - m0).norm())


def bounds(pb):
    dec = pg.decompose(pb["G"], pb["D"].gaussian())      # identity noise metric: the data ball is Euclidean
    res = dec.ellipsoid_intervals(pb["T"], a=pb["model_radius"], r=pb["data_radius"], datum=pb["d"], centre=pb["m0"])
    # prior-only bounds: <t, m0> +- a ||t|| (the ball's support function)
    tm0 = pb["T"](pb["m0"]).values
    tn = np.sqrt(np.diag((pb["T"] @ pb["T"].adjoint).matrix()))
    return res, tm0 - pb["model_radius"] * tn, tm0 + pb["model_radius"] * tn


if __name__ == "__main__":
    t0 = time.perf_counter()
    pb = build()
    res, plo, phi = bounds(pb)
    t1 = time.perf_counter()
    true = pb["T"](pb["m_bar"]).values
    lo, hi = np.asarray(res.lower), np.asarray(res.upper)
    print(f"data radius {pb['data_radius']:.4f}  model radius {pb['model_radius']:.4f}  time {t1 - t0:.2f} s")
    print(f"captured {np.sum((lo <= true) & (true <= hi))}/{N_P}; statuses {set(res.status_lower) | set(res.status_upper)}")
    for c, l, h, t in zip(pb["centres"], lo, hi, true):
        print(f"  x={c:.3f}  [{l:+.4f}, {h:+.4f}]  true {t:+.4f}")
    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(pb["centres"], lo, hi, alpha=0.25, label="admissible properties (pygeoinf0)")
        ax.fill_between(pb["centres"], plo, phi, alpha=0.1, color="k", label="model ball alone")
        ax.plot(pb["centres"], true, "rx", label="true properties")
        ax.legend(); ax.set_xlabel("target location"); ax.set_ylabel("property value")
        fig.savefig("dli_pygeoinf0.png", dpi=150, bbox_inches="tight")
