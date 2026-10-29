import pandas as pd
import pytest

from cfa.geo.crosswalk import crosswalk_report, match_districts, normalise_name

REF = pd.DataFrame(
    {
        "district_id": ["D1", "D2", "D3", "D4", "D5"],
        "district": ["Gurugram", "Prayagraj", "Bijapur", "Bijapur", "Ahmedabad"],
        "state": ["Haryana", "Uttar Pradesh", "Karnataka", "Chhattisgarh", "Gujarat"],
    }
)


def test_normalise_name_handles_noise():
    assert (
        normalise_name("  Ahmadabad (Urban) District ", {"ahmadabad": "ahmedabad"}) == "ahmedabad"
    )
    assert normalise_name("Jammu & Kashmir") == "jammu and kashmir"
    assert normalise_name(float("nan")) == ""
    assert normalise_name("Pūne") == "pune"


def test_exact_alias_and_fuzzy_matches():
    src = pd.DataFrame(
        {
            "district": ["Gurgaon", "ALLAHABAD", "Ahmedabaad", "Bijapur"],
            "state": ["Haryana", "Uttar Pradesh", "Gujarat", "Karnataka"],
        }
    )
    cw = match_districts(src, REF).set_index("source_district")
    assert cw.loc["Gurgaon", "district_id"] == "D1"
    assert cw.loc["ALLAHABAD", "district_id"] == "D2"
    assert cw.loc["Ahmedabaad", "method"] == "fuzzy" and cw.loc["Ahmedabaad", "district_id"] == "D5"
    # Same name in two states resolves by state, not by guesswork.
    assert cw.loc["Bijapur", "district_id"] == "D3"


def test_state_aliases_and_unknown_state():
    src = pd.DataFrame({"district": ["Gurugram", "Gurugram"], "state": ["HARYANA", "Atlantis"]})
    cw = match_districts(src, REF)
    assert cw.loc[0, "district_id"] == "D1"
    assert cw.loc[1, "method"] == "no_state_match" and pd.isna(cw.loc[1, "district_id"])


def test_weak_match_is_left_unmatched_not_guessed():
    src = pd.DataFrame({"district": ["Zzzzz"], "state": ["Gujarat"]})
    assert match_districts(src, REF).loc[0, "method"] == "unmatched"


def test_ambiguous_near_ties_are_not_guessed():
    ref = pd.DataFrame(
        {
            "district_id": ["A", "B"],
            "district": ["Nagpurr", "Nagpurs"],
            "state": ["Maharashtra"] * 2,
        }
    )
    src = pd.DataFrame({"district": ["Nagpur"], "state": ["Maharashtra"]})
    assert match_districts(src, ref).loc[0, "method"] == "ambiguous"


def test_manual_override_wins():
    src = pd.DataFrame({"district": ["Weird Name"], "state": ["Gujarat"]})
    cw = match_districts(src, REF, overrides={("gujarat", "weird name"): "D5"})
    assert cw.loc[0, "district_id"] == "D5" and cw.loc[0, "method"] == "manual"


def test_duplicate_targets_flagged_and_reported():
    src = pd.DataFrame({"district": ["Gurugram", "Gurgaon"], "state": ["Haryana"] * 2})
    cw = match_districts(src, REF)
    assert cw["duplicate_target"].all()
    rep = crosswalk_report(cw)
    assert rep["n_duplicate_targets"] == 2 and rep["match_rate"] == 1.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        match_districts(REF, REF, cutoff=0)
    with pytest.raises(ValueError):
        match_districts(REF, pd.concat([REF, REF]))
