"""Generate a small fake UK Biobank extract in the real column conventions.

style="f": ukbconv style  f.<field>.<instance>.<array>, id column f.eid
style="p": RAP style      p<field>_i<instance>_a<array> (p31, p34, p52 without instance), id eid
Also returns an Olink NPX long table (eid, ins_index, protein_id, result) + coding 143.
Truth (expected visit ages, events) is returned for assertions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABS = {30690: (5.7, 1.1), 30760: (1.45, 0.38), 30780: (3.5, 0.85), 30870: (1.7, 0.9), 30750: (36, 5),
        30740: (5.1, 1.0), 30710: (2.5, 3.0), 30700: (72, 15), 30020: (14, 1.2), 30600: (45, 2.6),
        30000: (6.8, 1.8), 30040: (91, 4.5), 30070: (13.4, 0.9), 30080: (250, 58), 21001: (27.4, 4.5),
        48: (90, 13), 30720: (0.9, 0.18)}


def col(style, field, inst=None, arr=None):
    if style == "f":
        return f"f.{field}.{inst or 0}.{arr or 0}"
    s = f"p{field}"
    if inst is not None:
        s += f"_i{inst}"
    if arr is not None:
        s += f"_a{arr}"
    return s


def make_fake_ukb(n=200, style="p", seed=0, n_proteins=20):
    rng = np.random.default_rng(seed)
    eid = np.arange(1000001, 1000001 + n)
    d = {"eid" if style == "p" else "f.eid": eid}
    yob = rng.integers(1937, 1970, n)
    mob = rng.integers(1, 13, n)
    d[col(style, 34, None if style == "p" else 0)] = yob
    d[col(style, 52, None if style == "p" else 0)] = mob
    d[col(style, 31, None if style == "p" else 0)] = rng.integers(0, 2, n)
    birth = pd.to_datetime(dict(year=yob, month=mob, day=15))
    base_date = pd.to_datetime("2006-03-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D")
    has1 = rng.uniform(size=n) < 0.3
    has2 = rng.uniform(size=n) < 0.2
    dates = {0: base_date,
             1: pd.Series(np.where(has1, base_date + pd.Timedelta(days=int(4.5 * 365)), pd.NaT)),
             2: pd.Series(np.where(has2, base_date + pd.Timedelta(days=int(9 * 365)), pd.NaT))}
    expected_age = {}
    for inst, dt in dates.items():
        dt = pd.to_datetime(pd.Series(dt))
        d[col(style, 53, inst, 0 if style == "p" else None)] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), None)
        age = (dt - birth).dt.days / 365.25
        d[col(style, 21003, inst, 0 if style == "p" else None)] = np.floor(age)
        expected_age[inst] = age.to_numpy()
        ok = dt.notna().to_numpy()
        for f, (mu, sd) in LABS.items():
            v = np.where(ok, np.abs(rng.normal(mu, sd, n)), np.nan)
            if f == 30750:
                v[rng.uniform(size=n) < 0.1] = np.nan
            d[col(style, f, inst, 0 if style == "p" else None)] = v
        for a in (0, 1):
            d[col(style, 4080, inst, a)] = np.where(ok, rng.normal(135, 18, n).round(), np.nan)
            d[col(style, 4079, inst, a)] = np.where(ok, rng.normal(80, 10, n).round(), np.nan)
        d[col(style, 46, inst, 0 if style == "p" else None)] = np.where(ok, rng.normal(28, 9, n).round(), np.nan)
        d[col(style, 47, inst, 0 if style == "p" else None)] = np.where(ok, rng.normal(30, 9, n).round(), np.nan)
        sm = np.where(ok, rng.choice([0, 1, 2, -3], n, p=[.5, .35, .1, .05]), np.nan)
        d[col(style, 20116, inst, 0 if style == "p" else None)] = sm
        d[col(style, 2443, inst, 0 if style == "p" else None)] = np.where(ok, rng.choice([0, 1, -1], n, p=[.9, .07, .03]), np.nan)
        for a in range(3):
            d[col(style, 6177, inst, a)] = np.where(ok & (a == 0), rng.choice([1, 2, -7], n), np.nan)
    died = rng.uniform(size=n) < 0.1
    ddate = base_date + pd.to_timedelta(rng.integers(365, 14 * 365, n), unit="D")
    d[col(style, 40000, 0, 0 if style == "p" else None)] = pd.Series(ddate.strftime("%Y-%m-%d")).where(died, None)
    d[col(style, 40001, 0, 0 if style == "p" else None)] = pd.Series(rng.choice(["I219", "C349", "J449"], n)).where(died, None)
    for a in range(4):
        has = rng.uniform(size=n) < 0.5
        d[col(style, 41270, None if style == "p" else 0, a)] = pd.Series(rng.choice(["I10", "E119", "I252", "C509"], n)).where(has, None)
        hd = base_date + pd.to_timedelta(rng.integers(-2000, 4000, n), unit="D")
        d[col(style, 41280, None if style == "p" else 0, a)] = pd.Series(hd.strftime("%Y-%m-%d")).where(has, None)
    main = pd.DataFrame(d)
    # olink long (instance 0 for 40% of people)
    pe = eid[rng.uniform(size=n) < 0.4]
    ol = pd.DataFrame([(e, 0, p + 1, rng.normal(0, 1)) for e in pe for p in range(n_proteins)],
                      columns=["eid", "ins_index", "protein_id", "result"])
    coding = pd.DataFrame({"coding": np.arange(1, n_proteins + 1),
                           "meaning": [f"GENE{p};Protein number {p}" for p in range(1, n_proteins + 1)]})
    truth = {"eid": eid, "expected_age": expected_age, "died": died, "death_date": ddate, "birth": birth,
             "has1": has1, "olink_eids": pe}
    return main, ol, coding, truth
