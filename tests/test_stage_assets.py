import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage_assets import take_valid, validate_messages, write_jsonl


class StageAssetsTest(unittest.TestCase):
    def test_validate_messages_normalizes_content(self):
        row = {
            "id": "x",
            "messages": [
                {"role": "user", "content": " Hej "},
                {"role": "assistant", "content": " Hej! "},
            ],
        }
        self.assertEqual(validate_messages(row)["messages"][0]["content"], "Hej")

    def test_take_valid_skips_bad_rows(self):
        rows = [
            {"id": "bad", "messages": []},
            {
                "id": "ok",
                "messages": [
                    {"role": "user", "content": "Fråga"},
                    {"role": "assistant", "content": "Svar"},
                ],
            },
        ]
        self.assertEqual([r["id"] for r in take_valid(rows, 1)], ["ok"])

    def test_jsonl_hash_is_stable(self):
        rows = [{"id": "x", "messages": [{"role": "user", "content": "å"}]}]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "x.jsonl"
            first = write_jsonl(target, rows)
            second = write_jsonl(target, rows)
            self.assertEqual(first, second)
            self.assertEqual(json.loads(target.read_text())["id"], "x")


if __name__ == "__main__":
    unittest.main()

