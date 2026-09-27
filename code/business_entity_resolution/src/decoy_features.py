"""Decoy-aware pair features: corruption-aware house numbers, extra name words, sort/ratio/JW similarities,
and context features computed within the candidate set (no country feature).

Inputs are aligned text dicts for the query side (S2/S3) and the S1 side with the normalised fields
name_core (legal forms and filler already removed), name_compact, addr_clean and numbers (filler
numbers such as PMB / PO Box already excluded, original order).

    pair_features(qt, st) -> dict of float32 arrays
    context_features(df)  -> adds per-query / per-S1 context columns to a pair table
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

TOKEN_MATCH_RATIO = 85
_NAN = np.float32(np.nan)


# ------------------------------------------------------------------ numbers
def main_number(numbers: str) -> str:
    """Return the main house number: the first number of the (filler-free) numbers field, zeros stripped ('' if none)."""
    if not numbers:
        return ""
    first = numbers.split()[0]
    return first.lstrip("0") or "0"


def num_relation(a: str, b: str) -> tuple:
    """Return (equal, compatible, conflict) for two zero-stripped main numbers ('' = missing -> all False).

    compatible = equal, or one is a prefix/suffix of the other (corruption noise such as 2007 -> 007 / 02007);
    conflict = both present and not compatible.
    """
    if not a or not b:
        return False, False, False
    equal = a == b
    compatible = equal or a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)
    return equal, compatible, not compatible


def number_features(q_numbers: list, s_numbers: list) -> dict:
    """Return num_equal / num_compatible / num_conflict / num_missing / num_edit / num_logdiff / any_shared."""
    n = len(q_numbers)
    out = {k: np.zeros(n, dtype=np.float32) for k in ("num_equal", "num_compatible", "num_conflict", "num_missing",
                                                       "any_shared_number")}
    out["num_edit"] = np.full(n, _NAN, dtype=np.float32)
    out["num_logdiff"] = np.full(n, _NAN, dtype=np.float32)
    for i, (qn, sn) in enumerate(zip(q_numbers, s_numbers)):
        a, b = main_number(qn), main_number(sn)
        eq, comp, conf = num_relation(a, b)
        out["num_equal"][i], out["num_compatible"][i], out["num_conflict"][i] = eq, comp, conf
        out["num_missing"][i] = (not a) + (not b)
        if a and b:
            out["num_edit"][i] = Levenshtein.distance(a, b)
            out["num_logdiff"][i] = np.log1p(abs(int(a[:12]) - int(b[:12])))
        if qn and sn:
            sq = {x.lstrip("0") or "0" for x in qn.split()}
            out["any_shared_number"][i] = bool(sq & {x.lstrip("0") or "0" for x in sn.split()})
    return out


# ------------------------------------------------------------------ names
def unmatched_tokens(a_tokens: list, b_tokens: list, min_ratio: int = TOKEN_MATCH_RATIO) -> list:
    """Return the tokens of ``a_tokens`` with no token in ``b_tokens`` at rapidfuzz ratio >= min_ratio."""
    if not b_tokens:
        return list(a_tokens)
    bset = set(b_tokens)
    out = []
    for t in a_tokens:
        if t in bset:
            continue
        best = process.extractOne(t, b_tokens, scorer=fuzz.ratio, score_cutoff=min_ratio)
        if best is None:
            out.append(t)
    return out


def extra_token_features(q_core: list, s_core: list) -> dict:
    """Return extra_tokens_q / extra_tokens_s1 (unmatched name_core tokens on each side)."""
    n = len(q_core)
    eq, es = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(q_core, s_core)):
        at, bt = a.split(), b.split()
        eq[i] = len(unmatched_tokens(at, bt))
        es[i] = len(unmatched_tokens(bt, at))
    return {"extra_tokens_q": eq, "extra_tokens_s1": es}


# ------------------------------------------------------------------ similarities
def _strip_digits(texts) -> list:
    """Return addresses without digits (the number is handled separately), whitespace collapsed."""
    return [" ".join("".join(ch for ch in t if not ch.isdigit()).replace(",", " ").split()) for t in texts]


def _sims(a, b, prefix: str) -> dict:
    """Return token_sort / ratio / token_set / jaro_winkler similarities (0-1) of aligned string lists."""
    out = {}
    for name, scorer, scale in (("tsort", fuzz.token_sort_ratio, 100), ("ratio", fuzz.ratio, 100),
                                ("tset", fuzz.token_set_ratio, 100),
                                ("jw", JaroWinkler.normalized_similarity, 1)):
        out[f"{prefix}_{name}"] = (process.cpdist(a, b, scorer=scorer, workers=-1) / scale).astype(np.float32)
    return out


def pair_features(qt: dict, st: dict) -> dict:
    """Return every pair feature for aligned query / S1 text dicts (object arrays of normalised fields)."""
    f = {}
    f.update(_sims(qt["name_core"], st["name_core"], "name"))
    f["name_compact_ratio"] = (process.cpdist(qt["name_compact"], st["name_compact"], scorer=fuzz.ratio,
                                              workers=-1) / 100).astype(np.float32)
    qa, sa = _strip_digits(qt["addr_clean"]), _strip_digits(st["addr_clean"])
    f.update(_sims(qa, sa, "addr"))
    empty = np.array([not x or not y for x, y in zip(qa, sa)])
    for k in [k for k in f if k.startswith("addr_")]:
        f[k][empty] = _NAN
    f["addr_empty_any"] = empty.astype(np.float32)
    f.update(number_features(list(qt["numbers"]), list(st["numbers"])))
    f.update(extra_token_features(list(qt["name_core"]), list(st["name_core"])))
    f["name_len_q"] = np.array([len(x.split()) for x in qt["name_core"]], dtype=np.float32)
    f["name_len_s1"] = np.array([len(x.split()) for x in st["name_core"]], dtype=np.float32)
    return f


def base_score(df: pd.DataFrame) -> np.ndarray:
    """Return a simple similarity used to rank candidates for the context features (name + address token_sort)."""
    return (0.5 * df["name_tsort"].to_numpy() + 0.5 * np.nan_to_num(df["addr_tsort"].to_numpy(), nan=0.0))


# ------------------------------------------------------------------ context
def context_features(df: pd.DataFrame, main_q: np.ndarray) -> pd.DataFrame:
    """Add context columns to a pair table (query_id, s1_id, pair features); ``main_q`` = query main numbers.

    q_best, q_margin (best minus this pair's base score for the top pair: best - second best), q_rank;
    s1_claims (queries whose rank-1 S1 is this S1), num_agree_major (1 / 0 / NaN: this query's main number
    compatible with the majority main number of the S1's other claimants), extra_vs_min (extra_tokens_q
    minus the minimum extra_tokens_q among the S1's claimants).
    """
    s = base_score(df)
    qid = df["query_id"].to_numpy()
    order = np.lexsort((df["s1_id"].to_numpy(), -s, qid))
    rank = np.empty(len(df), dtype=np.int32)
    qs = qid[order]
    start = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    rank[order] = np.arange(len(df)) - np.repeat(start, np.diff(np.r_[start, len(df)]))
    df["q_rank"] = (rank + 1).astype(np.float32)
    best = pd.Series(s).groupby(qid).transform("max").to_numpy()
    second = pd.Series(np.where(rank == 1, s, -np.inf)).groupby(qid).transform("max").to_numpy()
    df["q_best"] = best.astype(np.float32)
    df["q_gap_to_best"] = (best - s).astype(np.float32)
    df["q_margin"] = np.where(np.isfinite(second), best - second, best).astype(np.float32)
    df["base_score"] = s.astype(np.float32)

    claim = rank == 0
    cl = pd.DataFrame({"s1_id": df["s1_id"].to_numpy()[claim], "num": main_q[claim],
                       "extra": df["extra_tokens_q"].to_numpy()[claim], "query_id": qid[claim]})
    df["s1_claims"] = df["s1_id"].map(cl.groupby("s1_id").size()).fillna(0).astype(np.float32).to_numpy()
    df["extra_vs_min"] = (df["extra_tokens_q"].to_numpy()
                          - df["s1_id"].map(cl.groupby("s1_id")["extra"].min()).fillna(0).to_numpy()).astype(np.float32)
    # majority main number among claimants (excluding this query when it is itself a claimant)
    cln = cl[cl["num"] != ""]
    counts = cln.groupby(["s1_id", "num"]).size().rename("n").reset_index()
    major = counts.sort_values(["s1_id", "n", "num"], ascending=[True, False, True]).drop_duplicates("s1_id")
    major_num = df["s1_id"].map(major.set_index("s1_id")["num"]).fillna("").to_numpy()
    major_n = df["s1_id"].map(major.set_index("s1_id")["n"]).fillna(0).to_numpy()
    own = claim & (main_q == major_num)
    agree = np.full(len(df), np.nan, dtype=np.float32)
    for i in np.flatnonzero((major_num != "") & (main_q != "") & ~(own & (major_n <= 1))):
        agree[i] = float(num_relation(main_q[i], major_num[i])[1])
    df["num_agree_major"] = agree
    return df


# ------------------------------------------------------------------ v3: tagged address numbers
FLOOR_WORDS = frozenset({"floor", "flr", "fl", "floors"})
WORD_ORDINALS = {"ground": "0", "gr": "0", "grd": "0", "first": "1", "second": "2", "third": "3", "fourth": "4",
                 "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
_RE_ORD = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_RE_DIG = re.compile(r"\d+")
POSTAL_MIN_LEN = 5          # a postal-like token has at least this many digits ...
POSTAL_MIN_SHARE = 0.05     # ... and its length is a trailing-number length in >= this share of addresses


def learn_postal_lengths(addresses, min_len: int = POSTAL_MIN_LEN, min_share: float = POSTAL_MIN_SHARE) -> frozenset:
    """Return the digit lengths that behave like postal codes: the LAST token of the LAST address component is a
    pure number of that length (>= min_len) in at least ``min_share`` of the (non-empty) addresses. Data-driven and
    pooled over all records (no country logic); an empty set disables the postal tag."""
    counts, n = Counter(), 0
    for a in addresses:
        if not a:
            continue
        n += 1
        last = a.rsplit(",", 1)[-1].split()
        if last and last[-1].isdigit() and len(last[-1]) >= min_len:
            counts[len(last[-1])] += 1
    return frozenset(k for k, v in counts.items() if n and v / n >= min_share)


def tag_numbers(addr_clean: str, postal_lengths: frozenset = frozenset()) -> tuple:
    """Return (street, floor, postal) number tuples (zero-stripped, first-seen order) of a normalised address.

    floor  = ordinals ('2nd', '4th'), word ordinals next to a floor word ('first floor', 'gr flr') and numbers next
             to a floor word ('floor 3', 'fl 0', '3 floor');
    postal = a pure number in the last token position of the last component whose length is postal-like;
    street = every other digit run (house / plot / unit numbers; '646a' -> 646). Filler numbers (PMB, PO Box) are
             already removed from addr_clean by normalisation.
    """
    street, floor, postal = [], [], []
    comps = [c.split() for c in addr_clean.split(",")] if addr_clean else []
    for ci, toks in enumerate(comps):
        for i, t in enumerate(toks):
            prev_floor = i > 0 and toks[i - 1] in FLOOR_WORDS
            next_floor = i + 1 < len(toks) and toks[i + 1] in FLOOR_WORDS
            m = _RE_ORD.match(t)
            if m:
                floor.append(m.group(1).lstrip("0") or "0")
            elif t in WORD_ORDINALS and (next_floor or prev_floor):
                floor.append(WORD_ORDINALS[t])
            elif t.isdigit():
                v = t.lstrip("0") or "0"
                if prev_floor or next_floor:
                    floor.append(v)
                elif (postal_lengths and ci == len(comps) - 1 and i == len(toks) - 1
                      and len(t) in postal_lengths):
                    postal.append(v)
                else:
                    street.append(v)
            else:
                street.extend(d.lstrip("0") or "0" for d in _RE_DIG.findall(t))
    return tuple(dict.fromkeys(street)), tuple(dict.fromkeys(floor)), tuple(dict.fromkeys(postal))


def _compatible(a: str, b: str) -> bool:
    """Return True if two zero-stripped numbers are equal or one is a prefix / suffix of the other."""
    return num_relation(a, b)[1]


def tagged_number_features(q_tags: list, s_tags: list) -> dict:
    """Return v3 number features for aligned lists of tag_numbers() tuples (query side, S1 side).

    st_compat     query street numbers with a compatible S1 street number (zero / prefix / suffix rule)
    st_conflict   query street numbers with no compatible S1 street number (0 if the S1 has none), plus the same
                  count from the S1 side (st_conflict_s1)
    st_jaccard    exact Jaccard of the two street-number sets (NaN if both empty)
    st_main_*     relation of the FIRST street numbers (equal / compatible / conflict)
    floor_equal / floor_conflict, postal_equal / postal_conflict   (both sides tagged, sets intersect / disjoint)
    q_no_number / s1_no_number    no street number on that side
    """
    n = len(q_tags)
    names = ("st_compat", "st_conflict", "st_conflict_s1", "st_main_equal", "st_main_compat", "st_main_conflict",
             "floor_equal", "floor_conflict", "postal_equal", "postal_conflict", "q_no_number", "s1_no_number",
             "n_street_q", "n_street_s1")
    out = {k: np.zeros(n, dtype=np.float32) for k in names}
    out["st_jaccard"] = np.full(n, _NAN, dtype=np.float32)
    for i, ((qs, qf, qp), (ss, sf, sp)) in enumerate(zip(q_tags, s_tags)):
        out["n_street_q"][i], out["n_street_s1"][i] = len(qs), len(ss)
        out["q_no_number"][i], out["s1_no_number"][i] = not qs, not ss
        if qs and ss:
            comp = sum(any(_compatible(a, b) for b in ss) for a in qs)
            out["st_compat"][i] = comp
            out["st_conflict"][i] = len(qs) - comp
            out["st_conflict_s1"][i] = sum(not any(_compatible(b, a) for a in qs) for b in ss)
            sq, sss = set(qs), set(ss)
            out["st_jaccard"][i] = len(sq & sss) / len(sq | sss)
            eq, cp, cf = num_relation(qs[0], ss[0])
            out["st_main_equal"][i], out["st_main_compat"][i], out["st_main_conflict"][i] = eq, cp, cf
        if qf and sf:
            hit = bool(set(qf) & set(sf))
            out["floor_equal"][i], out["floor_conflict"][i] = hit, not hit
        if qp and sp:
            hit = bool(set(qp) & set(sp))
            out["postal_equal"][i], out["postal_conflict"][i] = hit, not hit
    return out


# ------------------------------------------------------------------ v3: extra words (for target encoding)
def extra_token_lists(q_core, s_core) -> tuple:
    """Return (extra words of the query, extra words of the S1) per pair as space-joined strings."""
    xq, xs = [], []
    for a, b in zip(q_core, s_core):
        at, bt = a.split(), b.split()
        xq.append(" ".join(unmatched_tokens(at, bt)))
        xs.append(" ".join(unmatched_tokens(bt, at)))
    return np.asarray(xq, dtype=object), np.asarray(xs, dtype=object)
