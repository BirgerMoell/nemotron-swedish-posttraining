import unittest

from scripts.evaluate_generations import apply_check


class GenerationChecksTest(unittest.TestCase):
    def test_exact_and_structured_checks(self):
        self.assertTrue(apply_check("42", {"type": "regex", "pattern": "^42$"}))
        self.assertTrue(
            apply_check('{"stad":"Göteborg","land":"Sverige"}', {"type": "json_keys", "keys": ["land", "stad"]})
        )
        self.assertTrue(apply_check("- ett\n- två\n- tre", {"type": "bullet_count", "count": 3}))
        self.assertFalse(apply_check("för många ord här", {"type": "max_words", "count": 3}))


if __name__ == "__main__":
    unittest.main()
