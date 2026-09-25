"""Pure-text normalisation rules for business names and addresses (plan Sections 3.5 and 6.1).

Every rule is decided by text patterns only, never by the country label, so it
applies unchanged to countries without training labels. This module imports no
pandas so worker processes stay light; the stage runner is src/normalize.py.

The rule tables (legal forms, address abbreviations, placeholders, ...) are
plain dicts/sets at the top so they are easy to extend or replace with
data-mined versions.
"""
import re
import unicodedata

from anyascii import anyascii

# ------------------------------------------------------------------ rule tables
# Canonical legal-form code -> surface phrases (lower-case, space-separated tokens,
# matched as whole-token sequences anywhere in the name, longest phrase first).
LEGAL_FORMS = {
    "PRIVATE_LIMITED": ["private limited", "private ltd", "pvt limited", "pvt ltd", "pvt", "private"],
    "LIMITED": ["limited", "ltd"],
    "LLC": ["llc"],
    "INC": ["inc", "incorporated"],
    "CORP": ["corp", "corporation"],
    "CO": ["co", "company", "cie"],
    "LLP": ["llp"],
    "LP": ["lp"],
    "PC": ["pc"],
    "PLLC": ["pllc"],
    "PLC": ["plc"],
    # French forms ("s a" and "s.a." become "sa" through dotted-initial joining)
    "SARL": ["sarl"],
    "SAS": ["sas"],
    "SASU": ["sasu"],
    "SA": ["sa"],
    "SCI": ["sci"],
    "EURL": ["eurl"],
    "SNC": ["snc"],
    # French family-business markers ("&" has already become "and")
    "FILS": ["and fils", "et fils"],
    "FRERES": ["and freres", "et freres"],
}

# Transliterated legal/business words -> English, applied only to names that contained
# non-Latin script (after anyascii). Mined from the data: top name_clean tokens among
# non-Latin names, fuzzy-matched to private/limited/company/trust/enterprises/llp
# (counts over train+test S2/S3 in comments). No variants of 'company' or 'trust' occur.
# Mapping to the English word (instead of straight to a legal code) keeps name_clean
# comparable with Latin names and lets extract_legal() assign the code.
TRANSLIT_WORDS = {
    "praivet": "private",       # 784,891
    "praibhet": "private",      # 91,555
    "piraivet": "private",      # 81,098
    "praivrr": "private",       # 45,739
    "limitet": "limited",       # 94,540
    "limirrd": "limited",       # 53,534
    "limtid": "limited",        # 16,099
    "elelpi": "llp",            # 95,652
    "emtrpraijej": "enterprises",   # 23,559
    "emtrpraiss": "enterprises",    # 3,343
    "emtrpraijes": "enterprises",   # 3,240
    "entrpiraics": "enterprises",   # 3,051
    "entarpraijes": "enterprises",  # 2,673
    "entrpraijhis": "enterprises",  # 2,620
    "enrrpraiss": "enterprises",    # 1,769
    "entrpraijes": "enterprises",   # 601
}
# Multi-token transliterations: 'pra li' is the Devanagari abbreviation of Pvt. Ltd. (134,723).
TRANSLIT_PHRASES = {("pra", "li"): ["private", "limited"]}

# Words dropped from name_core (after legal forms are removed).
NAME_FILLER = {"the", "and", "of", "et"}
# Titles dropped when they are the first token of a name.
NAME_TITLES = {"dr"}

# Unconditional address abbreviation expansions (context-dependent ones are in _expand_address_tokens).
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "pkwy": "parkway",
    "ln": "lane",
    "pl": "place",
    "imp": "impasse",
    "all": "allee",
    "che": "chemin", "ch": "chemin",
}
# Tokens that, following 'st'/'dr', mean the abbreviation is a street type.
STREET_TYPE_FOLLOWERS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
                         "apt", "apartment", "ste", "suite", "unit", "fl", "floor", "bldg", "building",
                         "rm", "room", "pmb", "box", "po", "lot", "spc", "trlr"}
# 'r' followed by one of these is 'rue'.
RUE_FOLLOWERS = {"de", "du", "des", "d", "la", "le"}

# Whole comma-segments that are placeholders, and tokens that are placeholders anywhere.
PLACEHOLDER_SEGMENTS = {"", "null", "n/a", "na", "none", "-", "--", "nan", "nil"}
PLACEHOLDER_TOKENS = {"null", "none", "nan"}

# Top-level domains recognised in website-style names (longest alternatives first).
DOMAIN_TLDS = ["co.in", "org.in", "net.in", "co.uk", "com", "info", "biz", "org", "net", "in", "fr",
               "io", "us", "eu"]

# Leetspeak folding applied inside name tokens that contain a letter.
LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "@": "a"})

# ------------------------------------------------------------------ compiled regexes
RE_DOTTED = re.compile(r"(?<![a-z0-9])[a-z](?:(?:\.\s*|\s+)[a-z](?![a-z0-9]))+\.?")
RE_NOT_LETTER = re.compile(r"[^a-z]")
RE_COOP = re.compile(r"\bco[\s.-]*op")
RE_BRACKET_ALPHA = re.compile(r"[(\[{][^()\[\]{}]*[a-z][^()\[\]{}]*[)\]}]")
RE_BRACKET_CHARS = re.compile(r"[()\[\]{}]")
RE_HASH_NUM = re.compile(r"#\s*\d+")
RE_LEADING_JUNK = re.compile(r"^[^a-z0-9]+")
RE_DOMAIN = re.compile(r"(?:^|(?<=[\s(]))(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*?)\.(?:"
                       + "|".join(re.escape(t) for t in DOMAIN_TLDS) + r")(?![a-z0-9])")
RE_NAME_TOKEN = re.compile(r"[a-z0-9@]+")
RE_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$")
RE_LETTER = re.compile(r"[a-z]")
# Filler numbers: only PMB / PO Box / box. '#<digits>' is NOT filler: on 400k true train pairs,
# the '#' number appears in the S1 address 96% (US) / 72% (India) of the time, whereas
# PMB / PO Box numbers appear 0% of the time.
RE_FILLER_NUM = re.compile(r"(?:\bpmb|\bpo\s*box|\bbox)\s*(?:no\b\.?)?\s*[#:.]?\s*(\d+)")
RE_NUM_SUFFIX = re.compile(r"(?<!\d)(\d+)\s*(?:bis|ter|quater)\b")
RE_SEGMENT_SPLIT = re.compile(r"[,;|\n]")
RE_ADDR_TOKEN = re.compile(r"[a-z0-9]+")
RE_DIGITS = re.compile(r"\d+")


def _build_legal_index(forms: dict) -> tuple:
    """Return ({token tuple: code}, max phrase length) for whole-token legal-form matching."""
    index = {}
    for code, phrases in forms.items():
        for phrase in phrases:
            index[tuple(phrase.split())] = code
    return index, max(len(k) for k in index)


LEGAL_INDEX, LEGAL_MAX_LEN = _build_legal_index(LEGAL_FORMS)


# ------------------------------------------------------------------ shared helpers
def _is_latin_or_common(ch: str) -> bool:
    """Return True if ``ch`` is a Latin-script letter (ASCII, Latin-1, Latin Extended ranges)."""
    o = ord(ch)
    return o < 0x250 or 0x1E00 <= o <= 0x1EFF or 0x2C60 <= o <= 0x2C7F or 0xA720 <= o <= 0xA7FF


def has_nonlatin(s: str) -> bool:
    """Return True if ``s`` contains a letter from a non-Latin script (Devanagari, Telugu, ...)."""
    return any(ch.isalpha() and not _is_latin_or_common(ch) for ch in s)


def to_ascii_lower(s: str) -> tuple:
    """NFKC, transliterate/accent-fold to ASCII (only when needed) and lowercase.

    Returns (text, nonlatin_flag). anyascii is only called for non-ASCII input.
    """
    nonlatin = False
    if not s.isascii():
        s = unicodedata.normalize("NFKC", s).replace("°", "o")  # 'N°' -> 'No' (anyascii gives 'deg')
        nonlatin = has_nonlatin(s)
        s = anyascii(s)
    return s.lower(), nonlatin


def _join_dotted(match: re.Match) -> str:
    """Return the letters of a dotted/spaced initials run ('l.c.s.w.' -> 'lcsw')."""
    return RE_NOT_LETTER.sub("", match.group())


def common_clean(s: str) -> str:
    """Cleaning shared by names and addresses: apostrophes, dotted initials, co-op, '&'/'+'."""
    s = s.replace("'", "").replace("`", "")
    s = RE_DOTTED.sub(_join_dotted, s)
    s = RE_COOP.sub("coop", s)
    return s.replace("&", " and ").replace("+", " and ")


def dedupe_consecutive(tokens: list) -> list:
    """Collapse runs of the same token ('family family bright' -> 'family bright')."""
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def unique(items) -> list:
    """Return the distinct items of ``items`` in first-seen order."""
    return list(dict.fromkeys(items))


def leet_fold(token: str) -> str:
    """Fold leetspeak digits inside tokens that contain a letter; keep numbers and ordinals."""
    if not RE_LETTER.search(token) or RE_ORDINAL.match(token):
        return token
    return token.translate(LEET_MAP)


def map_transliterated(tokens: list) -> list:
    """Replace mined transliterated words/phrases (e.g. 'praivet', 'pra li') with their English form."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        pair = tuple(tokens[i:i + 2])
        if pair in TRANSLIT_PHRASES:
            out.extend(TRANSLIT_PHRASES[pair])
            i += 2
        else:
            out.append(TRANSLIT_WORDS.get(tokens[i], tokens[i]))
            i += 1
    return out


def extract_legal(tokens: list) -> tuple:
    """Split tokens into (legal-form codes found, remaining tokens) by longest whole-token match."""
    codes, rest, i, n = [], [], 0, len(tokens)
    while i < n:
        for length in range(min(LEGAL_MAX_LEN, n - i), 0, -1):
            code = LEGAL_INDEX.get(tuple(tokens[i:i + length]))
            if code is not None:
                codes.append(code)
                i += length
                break
        else:
            rest.append(tokens[i])
            i += 1
    return codes, rest


# ------------------------------------------------------------------ names
def normalize_name(raw: str) -> tuple:
    """Normalise one business name.

    Returns (name_clean, legal_form, name_core, name_compact, initials,
    is_domain, name_nonlatin, name_core_fallback).
    """
    s, nonlatin = to_ascii_lower(raw)

    stem = None
    m = RE_DOMAIN.search(s)
    if m:
        stem = re.sub(r"[^a-z0-9]", "", m.group(1))
        s = f"{s[:m.start()]} {stem} {s[m.end():]}"

    s = common_clean(s)
    s = RE_BRACKET_ALPHA.sub(" ", s)
    s = RE_BRACKET_CHARS.sub(" ", s)
    s = RE_HASH_NUM.sub(" ", s)
    s = RE_LEADING_JUNK.sub("", s)
    tokens = RE_NAME_TOKEN.findall(s)
    if nonlatin:
        tokens = map_transliterated(tokens)
    tokens = dedupe_consecutive(tokens)
    clean_tokens = [t.replace("@", "a") for t in tokens]
    name_clean = " ".join(clean_tokens)

    fallback = False
    if stem:
        codes, core = [], [leet_fold(stem)]
    else:
        toks = [leet_fold(t) for t in tokens]
        if len(toks) > 1 and toks[0] in NAME_TITLES:
            toks = toks[1:]
        codes, rest = extract_legal(toks)
        core = [t for t in rest if t not in NAME_FILLER]
        if not core:  # the name was only legal/filler words: keep them rather than return nothing
            fallback = bool(toks)
            core = toks
    core = dedupe_consecutive(core)
    return (name_clean, "|".join(sorted(set(codes))), " ".join(core), "".join(core),
            "".join(t[0] for t in core), stem is not None, nonlatin, fallback)


# ------------------------------------------------------------------ addresses
def _expand_address_tokens(toks: list) -> list:
    """Expand abbreviations in one comma-segment using the neighbouring tokens for context."""
    out = []
    n = len(toks)
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else None
        if t in ("st", "dr"):
            if nxt is None or nxt in STREET_TYPE_FOLLOWERS:
                t = "street" if t == "st" else "drive"
            elif t == "st" and nxt.isalpha() and nxt != "no":
                t = "saint"
            # else: 'st no 8', 'st 5', 'dr ambedkar' -> unchanged
        elif t == "r":
            prv = toks[i - 1] if i > 0 else None
            if nxt is not None and (nxt in RUE_FOLLOWERS or (prv is not None and prv.isdigit() and nxt.isalpha())):
                t = "rue"
        else:
            t = ADDRESS_ABBREVIATIONS.get(t, t)
        out.append(t)
    return out


def normalize_address(raw: str) -> tuple:
    """Normalise one business address.

    Returns (addr_clean, numbers, filler_numbers, num_keys, addr_tokens, addr_empty);
    the list fields are space-joined strings.
    """
    s, _ = to_ascii_lower(raw)
    s = common_clean(s)
    s = RE_BRACKET_CHARS.sub(" ", s)  # keep bracket contents in addresses: '(east)', '(41)'
    filler = unique(f.lstrip("0") or "0" for f in RE_FILLER_NUM.findall(s))
    s = RE_FILLER_NUM.sub(" ", s)
    s = RE_NUM_SUFFIX.sub(r"\1", s)

    segments = []
    for seg in RE_SEGMENT_SPLIT.split(s):
        if seg.strip(" .") in PLACEHOLDER_SEGMENTS:
            continue
        toks = [t for t in RE_ADDR_TOKEN.findall(seg) if t not in PLACEHOLDER_TOKENS]
        toks = dedupe_consecutive(_expand_address_tokens(toks))
        if toks:
            segments.append(" ".join(toks))
    addr_clean = ", ".join(segments)

    numbers = unique(d.lstrip("0") or "0" for d in RE_DIGITS.findall(addr_clean))
    num_keys = unique(d.zfill(3)[-3:] for d in numbers)
    addr_tokens = unique(t for t in RE_ADDR_TOKEN.findall(addr_clean) if len(t) >= 2 and t.isalpha())
    return (addr_clean, " ".join(numbers), " ".join(filler), " ".join(num_keys),
            " ".join(addr_tokens), not segments)


# ------------------------------------------------------------------ batch entry point (for worker processes)
NAME_FIELDS = ("name_clean", "legal_form", "name_core", "name_compact", "initials",
               "is_domain", "name_nonlatin", "name_core_fallback")
ADDR_FIELDS = ("addr_clean", "numbers", "filler_numbers", "num_keys", "addr_tokens", "addr_empty")


def normalize_batch(args: tuple) -> dict:
    """Normalise a batch of (names, addresses); return {field: list} column-wise."""
    names, addrs = args
    cols = {}
    for fields, fn, values in ((NAME_FIELDS, normalize_name, names), (ADDR_FIELDS, normalize_address, addrs)):
        results = [fn(v) for v in values]
        for j, field in enumerate(fields):
            cols[field] = [r[j] for r in results]
    return cols
