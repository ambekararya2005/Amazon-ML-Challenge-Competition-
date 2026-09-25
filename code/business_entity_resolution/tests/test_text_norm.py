"""Unit tests for src/text_norm.py. Run from code/business_entity_resolution/: python -m unittest -v"""
import unittest

from src.text_norm import normalize_address, normalize_name


def name(raw: str) -> dict:
    """Return normalize_name output as a dict keyed by field name."""
    keys = ("clean", "legal", "core", "compact", "initials", "domain", "nonlatin", "fallback")
    return dict(zip(keys, normalize_name(raw)))


def addr(raw: str) -> dict:
    """Return normalize_address output as a dict keyed by field name."""
    keys = ("clean", "numbers", "filler", "keys", "tokens", "empty")
    return dict(zip(keys, normalize_address(raw)))


class TestNames(unittest.TestCase):
    """Name cleaning, legal forms, domains, leetspeak and transliteration."""

    def test_junk_prefix_and_leet(self):
        """'***' is stripped and leet digits are folded in name_core only."""
        r = name("*** Springda1e City Of")
        self.assertEqual(r["clean"], "springda1e city of")
        self.assertEqual(r["core"], "springdale city")

    def test_numbers_and_ordinals_not_leet_folded(self):
        """Pure numbers and ordinals keep their digits."""
        self.assertEqual(name("3rd Eye 1947 Traders")["core"], "3rd eye 1947 traders")

    def test_domain(self):
        """Website names become the TLD-free stem; '#digits' is removed."""
        r = name("springdalecity.com #39256")
        self.assertTrue(r["domain"])
        self.assertEqual((r["clean"], r["core"]), ("springdalecity", "springdalecity"))
        self.assertEqual(name("kimb1eolvas.com")["core"], "kimbleolvas")

    def test_dotted_initials_and_bracket_tags(self):
        """'L.C.S.W.' joins to 'lcsw'; '(PC)' and '[INC]' tags are removed."""
        self.assertEqual(name("Kimble, Olva S., L.C.S.W., (PC)")["clean"], "kimble olva s lcsw")
        r = name("CARDIOLOGY SAFE CARE INC [INC]")
        self.assertEqual((r["core"], r["legal"]), ("cardiology safe care", "INC"))

    def test_french_forms(self):
        """S.A.S / S.A. / SARL and '& Fils' markers are extracted."""
        self.assertEqual(name("Fractales Amis Groupe S.A.S")["legal"], "SAS")
        r = name("Europ & Frères Distribution S.A.")
        self.assertEqual((r["core"], r["legal"]), ("europ distribution", "FRERES|SA"))
        self.assertEqual(name("Grain & Fils")["legal"], "FILS")

    def test_india_forms_and_repeats(self):
        """Private Limited is one code; repeated words collapse; leading 'Dr' title dropped."""
        r = name("Secunderabad-Easy  Private Limited")
        self.assertEqual((r["core"], r["legal"]), ("secunderabad easy", "PRIVATE_LIMITED"))
        self.assertEqual(name("Family Family Bright Health,")["clean"], "family bright health")
        self.assertEqual(name("Dr Al  Tech")["core"], "al tech")

    def test_transliterated_legal_words(self):
        """Devanagari 'प्राइवेट लिमिटेड' -> private limited; 'प्रा. लि.' -> private limited."""
        r = name("अल टेक प्राइवेट लिमिटेड")
        self.assertTrue(r["nonlatin"])
        self.assertEqual((r["core"], r["legal"]), ("al tek", "PRIVATE_LIMITED"))
        self.assertEqual(name("अल टेक प्रा. लि.")["legal"], "PRIVATE_LIMITED")

    def test_accents_not_nonlatin(self):
        """Accented Latin is folded but is not flagged as non-Latin."""
        r = name("Engages Àrt Pharmacie SCI")
        self.assertEqual(r["core"], "engages art pharmacie")
        self.assertFalse(r["nonlatin"])

    def test_fallback_when_only_legal_words(self):
        """A name made only of legal words keeps them as core and flags the fallback."""
        r = name("The Company Ltd")
        self.assertTrue(r["fallback"])
        self.assertTrue(r["core"])


class TestAddresses(unittest.TestCase):
    """Placeholders, abbreviations with context, numbers and filler numbers."""

    def test_filler_and_placeholders(self):
        """PMB / PO Box numbers go to filler; NULL and N/A segments are dropped."""
        r = addr("607 VIRGINIA ST, PO BOX 2342, N/A, TERRELL, TX")
        self.assertEqual(r["clean"], "607 virginia street, terrell, tx")
        self.assertEqual((r["numbers"], r["filler"]), ("607", "2342"))
        self.assertEqual(addr("41 Chert Street, PMB 7452, Stephens")["filler"], "7452")

    def test_hash_numbers_are_real_numbers(self):
        """'#<digits>' is a house/unit number (kept in numbers), not filler; 'N°' reads as 'no'."""
        r = addr("##92219 Youngs River Road, Astoria")
        self.assertEqual((r["numbers"], r["filler"]), ("92219", ""))
        self.assertEqual(addr("# 45, 3rd Cross, Bangalore")["numbers"], "45 3")
        self.assertEqual(addr("N° 10 RUE DES POISSONNIERS")["clean"], "no 10 rue des poissonniers")

    def test_num_keys(self):
        """2007, 007 and 02007 share the key '007'."""
        for raw in ("2007 Robyn Road", "007 ROBYN RD", "02007 Robyn Road"):
            self.assertEqual(addr(raw)["keys"], "007")

    def test_st_rules(self):
        """'st' -> street at segment end / before unit words, saint before a word, kept before a number or 'no'."""
        self.assertEqual(addr("123 Main St, Boston")["clean"], "123 main street, boston")
        self.assertEqual(addr("123 Main St Apt 4, Boston")["clean"], "123 main street apt 4, boston")
        self.assertEqual(addr("12 St Pierre, Lyon")["clean"], "12 saint pierre, lyon")
        self.assertEqual(addr("S S Nagar, St No 8")["clean"], "ss nagar, st no 8")

    def test_dr_rules(self):
        """'dr' at segment end -> drive; before a name it stays."""
        self.assertEqual(addr("3634 PARK VISTA DR, TX")["clean"], "3634 park vista drive, tx")
        self.assertEqual(addr("Dr Ambedkar Road, Pune")["clean"], "dr ambedkar road, pune")

    def test_french(self):
        """R./R -> rue, AV -> avenue, bis dropped, bracketed number kept."""
        self.assertEqual(addr("63 R. DE DIEPPE, LILLE")["clean"], "63 rue de dieppe, lille")
        self.assertEqual(addr("18 R JEAN ZAY")["clean"], "18 rue jean zay")
        self.assertEqual(addr("77 AV LEON JOUHAUX")["clean"], "77 avenue leon jouhaux")
        r = addr("5 bis Rue Pierre Dignac")
        self.assertEqual((r["clean"], r["numbers"]), ("5 rue pierre dignac", "5"))
        self.assertEqual(addr("(41) Rue Des Thuyas")["numbers"], "41")

    def test_empty(self):
        """An empty or placeholder-only address is flagged empty."""
        self.assertTrue(addr("")["empty"])
        self.assertTrue(addr("NULL, N/A")["empty"])


if __name__ == "__main__":
    unittest.main()
