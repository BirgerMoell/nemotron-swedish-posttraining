import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_30b_smoke_is_one_step_across_eight_fsdp_ranks(self):
        config = json.loads((ROOT / "configs/lumi-s2-30b-post-smoke.json").read_text())
        training = config["training"]
        self.assertEqual(config["model"]["repo_id"], "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        self.assertEqual(training["expected_world_size"], 8)
        self.assertEqual(training["accepted_examples"], training["max_steps"] * 8)
        self.assertTrue(training["data_prevalidated"])
        self.assertTrue(training["cast_frozen_parameters_to_bf16"])
        self.assertEqual(training["max_length"], 256)
        self.assertNotIn("up_proj", training["target_modules"])
        self.assertNotIn("down_proj", training["target_modules"])

    def test_s1_global_example_count_matches_eight_rank_steps(self):
        config = json.loads((ROOT / "configs/lumi-s1-5k.json").read_text())
        training = config["training"]
        self.assertEqual(training["accepted_examples"], training["max_steps"] * 8)
        self.assertGreaterEqual(training["min_masked_prompt_tokens"], 16)
        self.assertGreaterEqual(training["min_supervised_tokens"], 32)

    def test_p1_global_example_count_matches_sixty_four_rank_steps(self):
        config = json.loads((ROOT / "configs/lumi-p1-100k.json").read_text())
        training = config["training"]
        self.assertEqual(training["expected_world_size"], 64)
        self.assertEqual(training["accepted_examples"], training["max_steps"] * 64)
        self.assertTrue(training["data_prevalidated"])
        self.assertIn("in_proj", training["target_modules"])
        self.assertIn("out_proj", training["target_modules"])


if __name__ == "__main__":
    unittest.main()
