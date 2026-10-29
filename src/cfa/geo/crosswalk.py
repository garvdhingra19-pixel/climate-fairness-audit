"""District-name crosswalk: link survey/station tables to a reference district list.

Names differ across sources (spelling, renames, 'District' suffixes), so matching is done
within state, exact first, then fuzzy. Ambiguous or weak matches are left UNMATCHED rather
than guessed, and every row records how it was matched so the result is auditable.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from difflib import SequenceMatcher

import pandas as pd

# Small, extensible alias tables (normalised form -> canonical normalised form).
DISTRICT_ALIASES = {
    "gurgaon": "gurugram",
    "allahabad": "prayagraj",
    "faizabad": "ayodhya",
    "mahbubnagar": "mahabubnagar",
    "ahmadabad": "ahmedabad",
}
STATE_ALIASES = {
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "nct of delhi": "delhi",
    "jammu kashmir": "jammu and kashmir",
    "andaman nicobar islands": "andaman and nicobar islands",
}
_NOISE = re.compile(r"\b(district|dist|zila)\b")


def normalise_name(name: object, aliases: Mapping[str, str] | None = None) -> str:
    """Lowercase, strip accents/punctuation/'district', collapse spaces, apply aliases."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"\(.*?\)", " ", text.lower().replace("&", " and "))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", _NOISE.sub(" ", text)).strip()
    return (aliases or {}).get(text, text)


def _norm_state(name: object) -> str:
    return normalise_name(name, STATE_ALIASES)


def _norm_district(name: object) -> str:
    return normalise_name(name, DISTRICT_ALIASES)


def match_districts(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    source_district_col: str = "district",
    source_state_col: str = "state",
    ref_district_col: str = "district",
    ref_state_col: str = "state",
    ref_id_col: str = "district_id",
    cutoff: float = 0.88,
    ambiguity_margin: float = 0.02,
    overrides: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Return one row per source row: district_id (NaN if unmatched), method, score.

    overrides maps (normalised_state, normalised_district) -> district_id for manual fixes.
    """
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must be in (0, 1].")
    if reference[ref_id_col].duplicated().any():
        raise ValueError("reference has duplicate district ids.")
    ref = reference[[ref_id_col, ref_district_col, ref_state_col]].copy()
    ref["_d"] = ref[ref_district_col].map(_norm_district)
    ref["_s"] = ref[ref_state_col].map(_norm_state)
    by_state = dict(iter(ref.groupby("_s")))

    rows = []
    for idx, rec in source.iterrows():
        d, s = _norm_district(rec[source_district_col]), _norm_state(rec[source_state_col])
        out = {
            "source_index": idx,
            "source_district": rec[source_district_col],
            "source_state": rec[source_state_col],
            "district_id": None,
            "matched_name": None,
            "method": "unmatched",
            "score": 0.0,
        }
        cands = by_state.get(s)
        if overrides and (s, d) in overrides:
            out.update(district_id=overrides[(s, d)], method="manual", score=1.0)
        elif cands is None or not d:
            out["method"] = "no_state_match" if cands is None else "empty_name"
        else:
            exact = cands[cands["_d"] == d]
            if len(exact) == 1:
                out.update(district_id=exact[ref_id_col].iloc[0], method="exact", score=1.0)
            elif len(exact) > 1:
                out["method"] = "ambiguous"
            else:
                scored = sorted(
                    (
                        (SequenceMatcher(None, d, c).ratio(), i)
                        for c, i in zip(cands["_d"], cands[ref_id_col], strict=True)
                    ),
                    reverse=True,
                )
                best = scored[0]
                if best[0] >= cutoff and (
                    len(scored) == 1 or best[0] - scored[1][0] > ambiguity_margin
                ):
                    out.update(district_id=best[1], method="fuzzy", score=round(best[0], 4))
                elif best[0] >= cutoff:
                    out["method"] = "ambiguous"
        if out["district_id"] is not None:
            names = ref.loc[ref[ref_id_col] == out["district_id"], ref_district_col]
            out["matched_name"] = names.iloc[0] if len(names) else None
        rows.append(out)

    cw = pd.DataFrame(rows)
    matched = cw["district_id"].notna()
    cw["duplicate_target"] = matched & cw["district_id"].duplicated(keep=False)
    return cw


def crosswalk_report(cw: pd.DataFrame) -> dict[str, object]:
    """Counts to put in a data-quality log: coverage, methods, duplicates (e.g. district splits)."""
    n = len(cw)
    return {
        "n_source": n,
        "n_matched": int(cw["district_id"].notna().sum()),
        "match_rate": float(cw["district_id"].notna().mean()) if n else 0.0,
        "by_method": cw["method"].value_counts().to_dict(),
        "n_duplicate_targets": int(cw["duplicate_target"].sum()),
        "unmatched": cw.loc[cw["district_id"].isna(), ["source_district", "source_state"]].to_dict(
            "records"
        ),
    }
