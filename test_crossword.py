import os
import unittest

import crossword as cw


class CrosswordTests(unittest.TestCase):
    def test_patterns(self):
        self.assertTrue(cw.matches("crane", "?R..E"))
        self.assertFalse(cw.matches("raven", "?R..E"))
        self.assertFalse(cw.matches("two words", ""))
        with self.assertRaises(ValueError):
            cw.matches("crane", "*")


@unittest.skipUnless(os.getenv("CROSSWORD_TEST_MODEL"), "Set CROSSWORD_TEST_MODEL to run a real checkpoint.")
class InterventionTests(unittest.TestCase):
    def test_solving_with_a_second_meaning(self):
        import engine
        name = os.environ["CROSSWORD_TEST_MODEL"]
        rows, _ = engine.solve_clue(name, "Marsh bird", "?R??E", "crane, heron", "hoist, pulley")
        self.assertEqual([row[0] for row in rows], ["crane"])
        self.assertEqual(len(rows[0]), 3)
        rows, _ = engine.solve_clue(name, "Marsh bird", "?R??E", "crane, heron")
        self.assertEqual(len(rows[0]), 2)

    def test_single_token_transplant_and_neutral_recovery(self):
        import engine
        name = os.environ["CROSSWORD_TEST_MODEL"]
        engine.load_model(name)
        lm = engine.reader(name)
        # Equal-length single-token slots leave all later positions aligned.
        word = next((w for w in ("tree", "bird", "fish") if len(lm.encode(lm.body(w, "A or B?"), w)[0])
                     == len(lm.encode(lm.body("thing", "A or B?"), "thing")[0])), None)
        self.assertIsNotNone(word)
        body = lm.body("thing", "A or B?")
        before = lm.margins([body])[0]
        patched = lm.margins([body], lm.prototype([word], 4))[0]
        direct = lm.margins([lm.body(word, "A or B?")])[0]
        neutral = lm.margins([body], lm.prototype(["thing"], 4))[0]
        self.assertAlmostEqual(patched, direct, delta=.06)
        self.assertAlmostEqual(neutral, before, delta=.06)
        self.assertAlmostEqual(lm.margins([body])[0], before, delta=.001)


if __name__ == "__main__":
    unittest.main()
