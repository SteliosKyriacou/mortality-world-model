import numpy as np
import pandas as pd
import pytest

from mwm.data.ukb import load_ukb, parse_column, read_main, ALL_FIELDS
from ukb_fake import make_fake_ukb


def test_parse_column():
    assert parse_column("f.4080.0.1") == (4080, 0, 1)
    assert parse_column("p4080_i2_a1") == (4080, 2, 1)
    assert parse_column("p31") == (31, 0, 0)
    assert parse_column("p53_i1") == (53, 1, 0)
    assert parse_column("eid") is None


@pytest.mark.parametrize("style", ["f", "p"])
def test_load_fake_ukb(style, tmp_path):
    main, ol, coding, truth = make_fake_ukb(n=150, style=style, seed=1)
    f = tmp_path / "ukb.csv"
    main.to_csv(f, index=False)
    df = read_main(f, fields=ALL_FIELDS)
    res = load_ukb(df, ol, coding)
    L, O = res.long, res.outcomes
    assert set(L.columns) == {"person_id", "cohort", "visit_age", "feature", "value", "unit"}
    # visit ages match expected fractional ages (to ~1 day + mid-month birth convention)
    v0 = res.visits[res.visits.instance == 0].set_index("person_id")["visit_age"]
    exp = pd.Series(truth["expected_age"][0], index=truth["eid"].astype(str))
    assert np.allclose(v0.reindex(exp.index), exp, atol=0.01)
    # repeat visits only where generated
    n1 = res.visits[res.visits.instance == 1].person_id.nunique()
    assert n1 == truth["has1"].sum()
    # SBP is mean of the two automated readings
    sbp0 = main[[c for c in main.columns if c.startswith(("f.4080.0.", "p4080_i0_"))]].mean(1)
    got = L[(L.feature == "sbp")].groupby("person_id").first()  # baseline is first by age
    assert np.allclose(got.loc[str(truth["eid"][0]), "value"], sbp0.iloc[0])
    # smoking code -3 removed
    assert (L.loc[L.feature == "smoking", "value"] >= 0).all()
    # grip is max of left/right
    assert "grip" in set(L.feature)
    # deaths
    assert O.event.sum() == truth["died"].sum()
    assert set(O.loc[O.event == 1, "cause"]) <= {"cvd", "cancer", "other"}
    assert (O.age_at_death_or_censor >= O.age_at_baseline).all()
    # olink present for the right people, unit NPX, gene names from coding 143
    olk = L[L.unit == "NPX"]
    assert set(olk.person_id) == set(truth["olink_eids"].astype(str))
    assert olk.feature.str.startswith("olink_gene").all()
    # HES aligned code/date
    assert {"person_id", "icd10", "age_at_event"} <= set(res.hes.columns)
    assert res.hes.age_at_event.notna().all()
    # medication flags
    assert {"bp_med", "lipid_med"} <= set(L.feature)
