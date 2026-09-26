"""Geographic units for the geo-dense validation benchmark (data-driven, no country-specific rules).

A unit is (country, city key). The city key comes from the comma-separated components of the
normalised address (addr_clean):
  * components containing a digit are street/number parts and are ignored;
  * generic tokens (tokens occurring in many distinct S1 components of the country, e.g. "city",
    "county") are dropped from a component to form its key;
  * region components (keys that are the most frequent key of the S1 record in most S1 records
    containing them, e.g. state names) are ignored;
  * the city vocabulary is every remaining key seen in >= VOCAB_MIN_COUNT S1 records.
An S1 record's unit is its most frequent vocabulary key; fallbacks: a postal-code-like component
(a component that is only 5-6 digits), else the street name (digits removed) of the first
component containing a digit. S2/S3 records carry ALL their vocabulary keys; a record belongs to
the benchmark if any of its keys is a sampled unit (keeps true pairs together when a city is
written in several ways).

Sampling and folds use a stable hash of "country|unit" (md5), never Python's hash().
"""
import hashlib
import re
from collections import Counter

GENERIC_MIN_COMPONENTS = 150      # token in >= this many distinct S1 components -> generic
REGION_MIN_COUNT = 500            # regions must be frequent
REGION_TOP_SHARE = 0.5            # ... and the record's top key in >= this share of their records
VOCAB_MIN_COUNT = 2
SAMPLE_MOD, SAMPLE_KEEP = 5, 0    # hash % 5 == 0 -> sampled (~20%)
N_FOLDS = 5

_NON_ALPHA = re.compile(r"[^a-z ]+")
_SPACES = re.compile(r"\s+")
_POSTAL = re.compile(r"^\d{5,6}$")
_DIGIT = re.compile(r"\d")


def split_components(addr_clean: str) -> tuple:
    """Return (text components without digits, raw components with digits) of a normalised address."""
    words, numbered = [], []
    for part in addr_clean.split(","):
        part = part.strip()
        if not part:
            continue
        if _DIGIT.search(part):
            numbered.append(part)
        else:
            clean = _SPACES.sub(" ", _NON_ALPHA.sub(" ", part)).strip()
            if clean:
                words.append(clean)
    return words, numbered


class Gazetteer:
    """Per-country city vocabulary fitted on S1 addresses."""

    def __init__(self, s1_addresses):
        """Fit generic tokens, region keys and the city vocabulary from S1 normalised addresses."""
        comps = [split_components(a)[0] for a in s1_addresses]
        tok_components = Counter()
        for c in set(x for cs in comps for x in cs):
            tok_components.update(set(c.split()))
        self.generic = {t for t, n in tok_components.items() if n >= GENERIC_MIN_COMPONENTS}
        keys = [self._keys(cs) for cs in comps]
        freq = Counter(k for ks in keys for k in set(ks))
        top = Counter(max(set(ks), key=lambda k: (freq[k], k)) for ks in keys if ks)
        self.regions = {k for k, n in freq.items() if n >= REGION_MIN_COUNT and top[k] >= REGION_TOP_SHARE * n}
        self.freq = {k: n for k, n in freq.items() if n >= VOCAB_MIN_COUNT and k not in self.regions}

    def _key(self, component: str) -> str:
        """Return the component with generic tokens removed (the component itself if nothing is left)."""
        kept = [t for t in component.split() if t not in self.generic]
        return " ".join(kept) if kept else component

    def _keys(self, components: list) -> list:
        """Return the keys of a list of components."""
        return [self._key(c) for c in components]

    def city_keys(self, addr_clean: str) -> list:
        """Return the record's vocabulary keys, most frequent first (deduplicated)."""
        words, _ = split_components(addr_clean)
        ks = {k for k in self._keys(words) if k in self.freq}
        return sorted(ks, key=lambda k: (-self.freq[k], k))

    def fallback_key(self, addr_clean: str) -> str:
        """Return a postal-code or street-name key for a record without a vocabulary key ('' if none)."""
        words, numbered = split_components(addr_clean)
        for part in numbered:
            if _POSTAL.match(part.replace(" ", "")):
                return "postal:" + part.replace(" ", "")
        for part in numbered:
            street = _SPACES.sub(" ", _NON_ALPHA.sub(" ", part)).strip()
            if street:
                return "street:" + street
        return "street:" + (words[0] if words else "")

    def unit(self, addr_clean: str) -> str:
        """Return the record's primary unit key."""
        ks = self.city_keys(addr_clean)
        return ks[0] if ks else self.fallback_key(addr_clean)

    def all_units(self, addr_clean: str) -> list:
        """Return every unit key the record belongs to (vocabulary keys, else the fallback key)."""
        return self.city_keys(addr_clean) or [self.fallback_key(addr_clean)]


def unit_hash(country: str, unit: str) -> int:
    """Return a stable 64-bit hash of (country, unit)."""
    return int.from_bytes(hashlib.md5(f"{country}|{unit}".encode("utf-8")).digest()[:8], "little")


def sampled(h: int) -> bool:
    """Return True if the unit with hash ``h`` is in the ~20% benchmark sample."""
    return h % SAMPLE_MOD == SAMPLE_KEEP


def fold_of(h: int) -> int:
    """Return the fold (0..N_FOLDS-1) of a unit hash; fold 0 is the held-out benchmark."""
    return (h // SAMPLE_MOD) % N_FOLDS
