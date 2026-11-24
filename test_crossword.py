import os
import unittest

import crossword as cw


class CrosswordTests(unittest.TestCase):
    def test_solution_entries_finish_each_grid(self):
        for puzzle, (rows, _) in cw.PUZZLES.items():
            fills = {}
            for key, (_, cells) in cw.slots(puzzle).items():
                fills = cw.enter(puzzle, key, fills, "".join(rows[y][x] for y, x in cells))
            self.assertIn("Solved!", cw.check(puzzle, fills))

    def test_crossings_and_clear(self):
        puzzle = "Pocket edition"
        fills = cw.enter(puzzle, "1A", {}, "tan")
        self.assertEqual(cw.pattern(puzzle, "3D", fills), "N??")
        with self.assertRaises(ValueError):
            cw.enter(puzzle, "3D", fills, "cat")
        fills = cw.enter(puzzle, "3D", fills, "net")
        del fills["1A"]
        self.assertEqual(cw.pattern(puzzle, "1A", fills), "??N")
        self.assertEqual(cw.pattern(puzzle, "3D", fills), "???")

    def test_patterns(self):
        self.assertTrue(cw.matches("crane", "?R..E"))
        self.assertFalse(cw.matches("raven", "?R..E"))
        self.assertFalse(cw.matches("two words", ""))
        with self.assertRaises(ValueError):
            cw.matches("crane", "*")


@unittest.skipUnless(os.getenv("CROSSWORD_TEST_MODEL"), "Set CROSSWORD_TEST_MODEL to run a real checkpoint.")
class InterventionTests(unittest.TestCase):
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
