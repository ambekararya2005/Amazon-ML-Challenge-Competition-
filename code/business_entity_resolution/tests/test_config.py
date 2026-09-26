"""Unit tests for path resolution (CLI > env > default, Kaggle data auto-detect). Run: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config, io_utils


class TestResolvePaths(unittest.TestCase):
    """Order of precedence for the four roots."""

    def test_defaults_unchanged(self):
        """With no flags or env vars the roots are the repo layout."""
        with mock.patch.object(config, "find_data_root", return_value=None):
            p = config.resolve_paths([], {})
        self.assertEqual(p["data"], config.REPO_ROOT / "student_resource" / "dataset")
        self.assertEqual(p["cache"], config.REPO_ROOT / "cache")
        self.assertEqual(p["output"], config.REPO_ROOT / "output")
        self.assertEqual(p["log"], config.REPO_ROOT / "logs")

    def test_env_then_cli(self):
        """Env vars override defaults, legacy BER_* vars are honoured, CLI flags override env vars."""
        env = {"CACHE_ROOT": "/e/cache", "BER_LOG_DIR": "/legacy/logs", "OUTPUT_ROOT": "/e/out"}
        p = config.resolve_paths(["--stage", "bench", "--output-root", "/cli/out"], env)
        self.assertEqual(p["cache"], Path("/e/cache"))
        self.assertEqual(p["log"], Path("/legacy/logs"))
        self.assertEqual(p["output"], Path("/cli/out"))

    def test_explicit_data_root_skips_autodetect(self):
        """An explicit DATA_ROOT is used even if it has no data yet (no search)."""
        with mock.patch.object(config, "find_data_root", side_effect=AssertionError("searched")):
            p = config.resolve_paths([], {"DATA_ROOT": "/nowhere"})
        self.assertEqual(p["data"], Path("/nowhere"))


class TestFindDataRoot(unittest.TestCase):
    """Auto-detection of the dataset folder."""

    def test_nested_and_flat_layouts(self):
        """<root>/train/train_source1.tsv gives <root>; a flat folder gives itself; none gives None."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertIsNone(config.find_data_root(tmp))
            nested = tmp / "a" / "amlc2026-data" / "dataset"
            (nested / "train").mkdir(parents=True)
            (nested / "train" / "train_source1.tsv").write_text("x", encoding="utf-8")
            self.assertEqual(config.find_data_root(tmp), nested)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "flat").mkdir()
            (tmp / "flat" / "train_source1.tsv").write_text("x", encoding="utf-8")
            self.assertEqual(config.find_data_root(tmp), tmp / "flat")
        self.assertIsNone(config.find_data_root(Path(tempfile.gettempdir()) / "does_not_exist_amlc"))

    def test_data_file_flat_fallback(self):
        """io_utils.data_file prefers <data>/<split>/<name>, falls back to <data>/<name>."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(io_utils, "DATA_DIR", tmp):
                self.assertEqual(io_utils.source_tsv_path("test", 2), tmp / "test" / "test_source2.tsv")
                (tmp / "test_source2.tsv").write_text("x", encoding="utf-8")
                self.assertEqual(io_utils.source_tsv_path("test", 2), tmp / "test_source2.tsv")
                (tmp / "test").mkdir()
                (tmp / "test" / "test_source2.tsv").write_text("x", encoding="utf-8")
                self.assertEqual(io_utils.source_tsv_path("test", 2), tmp / "test" / "test_source2.tsv")


class TestCpu(unittest.TestCase):
    """CPU count helpers."""

    def test_cpu_count_and_env_int(self):
        """cpu_count is positive; env_int reads overrides."""
        self.assertGreaterEqual(config.cpu_count(), 1)
        with mock.patch.dict("os.environ", {"N_WORKERS": "3"}):
            self.assertEqual(config.env_int("N_WORKERS", 9), 3)
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(config.env_int("AMLC_UNSET_VAR_XYZ", 9), 9)


if __name__ == "__main__":
    unittest.main()
