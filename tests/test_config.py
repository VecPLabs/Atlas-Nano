import unittest
from pathlib import Path

from atlas_nano.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_training_gauntlet_is_bundled(self):
        config = load_config("does-not-exist.yaml")
        self.assertTrue(Path(config.train.gauntlet).is_file())
        self.assertEqual(config.train.gauntlet, config.cache.gauntlet)
        self.assertEqual(config.train.gauntlet, config.sign_check.gauntlet)

    def test_mixed_provenance_benchmark_has_no_implicit_default(self):
        config = load_config("does-not-exist.yaml")
        self.assertIsNone(config.benchmark.gauntlet)


if __name__ == "__main__":
    unittest.main()
