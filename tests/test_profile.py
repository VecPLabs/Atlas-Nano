import json
import tempfile
import unittest
from pathlib import Path

from atlas_nano.profile import (
    ProfileError, RuntimeModel, assert_compatible, load_profile, verify_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "qwen3-4b-signcheck-v1" / "profile.json"


class ProfileTests(unittest.TestCase):
    def test_release_profile_is_valid(self):
        profile = load_profile(PROFILE)
        self.assertEqual(profile["evaluation"]["split"], "calibration")

    def test_matching_runtime_is_compatible(self):
        profile = load_profile(PROFILE)
        assert_compatible(profile, RuntimeModel(
            name="Qwen/Qwen3-4B",
            architecture="Qwen3ForCausalLM",
            hidden_dim=2560,
        ))

    def test_release_artifact_checksums_match(self):
        profile = load_profile(PROFILE)
        verify_artifacts(profile, PROFILE.parent)

    def test_profile_matches_exported_metadata(self):
        profile = load_profile(PROFILE)
        metadata = json.loads(
            (PROFILE.parent / "model_safety_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["safety.model_name"], profile["base_model"])
        self.assertEqual(metadata["safety.extraction_layer"], profile["extraction"]["layer"])
        self.assertEqual(metadata["safety.hidden_dim"], profile["extraction"]["hidden_dim"])
        self.assertAlmostEqual(metadata["safety.threshold"], profile["decision"]["threshold"])
        self.assertAlmostEqual(
            metadata["safety.calibration_f1"], profile["evaluation"]["metrics"]["f1"]
        )

    def test_mismatched_runtimes_are_rejected(self):
        profile = load_profile(PROFILE)
        runtimes = [
            RuntimeModel("meta-llama/Llama-3.2-3B", "Qwen3ForCausalLM", hidden_dim=2560),
            RuntimeModel("Qwen/Qwen3-4B", "LlamaForCausalLM", hidden_dim=2560),
            RuntimeModel("Qwen/Qwen3-4B", "Qwen3ForCausalLM", hidden_dim=4096),
        ]
        for runtime in runtimes:
            with self.subTest(runtime=runtime):
                with self.assertRaisesRegex(ProfileError, "incompatible safety profile"):
                    assert_compatible(profile, runtime)

    def test_missing_contract_field_is_rejected(self):
        data = json.loads(PROFILE.read_text(encoding="utf-8"))
        del data["extraction"]["hidden_dim"]
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_profile = Path(temp_dir) / "profile.json"
            bad_profile.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "extraction.hidden_dim"):
                load_profile(bad_profile)

    def test_unknown_schema_version_is_rejected(self):
        data = json.loads(PROFILE.read_text(encoding="utf-8"))
        data["schema_version"] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_profile = Path(temp_dir) / "profile.json"
            bad_profile.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "unsupported schema_version"):
                load_profile(bad_profile)


if __name__ == "__main__":
    unittest.main()
