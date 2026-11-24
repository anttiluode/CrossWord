"""Puzzle rules. The model only receives the clue and public crossing letters."""
import html
import re

PUZZLES = {
    "Pocket edition": (
        ["TAN", "ORE", "WET"],
        {"1A": "Summer shade acquired at the beach", "4A": "What a mine might yield",
         "5A": "Still dripping", "1D": "Pull a stranded car", "2D": "Verb after “we”",
         "3D": "What a butterfly catcher carries"},
    ),
    "Double meanings": (
        ["A#B#B", "R#E#E", "CRANE", "##R##", "##D##"],
        {"1D": "A bow's shape", "2D": "Chin covering", "3D": "Visitor to a flower",
         "4A": "One stands in a marsh, another lifts steel"},
    ),
    "The word square": (
        ["HEART", "EMBER", "ABUSE", "RESIN", "TREND"],
        {"1A": "The suit with a pulse?", "6A": "Something still glowing after a fire",
         "7A": "Misuse", "8A": "Sticky substance from a pine", "9A": "Direction in fashion",
         "1D": "Symbol on a valentine", "2D": "A coal that could reignite",
         "3D": "Treat cruelly", "4D": "Amber began as this", "5D": "A pattern in the data"},
    ),
}


def slots(puzzle):
    rows, clues = PUZZLES[puzzle]
    found = {}
    number = 0
    for r, row in enumerate(rows):
        for c, letter in enumerate(row):
            if letter == "#":
                continue
            across = (c == 0 or row[c-1] == "#") and c+1 < len(row) and row[c+1] != "#"
            down = (r == 0 or rows[r-1][c] == "#") and r+1 < len(rows) and rows[r+1][c] != "#"
            if across or down:
                number += 1
            for starts, axis, dr, dc in ((across, "A", 0, 1), (down, "D", 1, 0)):
                if starts:
                    y, x = r, c
                    cells = []
                    while y < len(rows) and x < len(row) and rows[y][x] != "#":
                        cells.append((y, x))
                        y, x = y + dr, x + dc
                    key = f"{number}{axis}"
                    found[key] = (clues[key], cells)
    return found


def letters(puzzle, fills, ignore=None):
    entries = slots(puzzle)
    return {cell: ch for key, word in fills.items() if key != ignore
            for cell, ch in zip(entries[key][1], word)}


def pattern(puzzle, key, fills):
    known = letters(puzzle, fills, ignore=key)
    return "".join(known.get(cell, "?") for cell in slots(puzzle)[key][1])


def enter(puzzle, key, fills, word):
    word = word.strip().upper()
    mask = pattern(puzzle, key, fills)
    if not re.fullmatch(r"[A-Z]+", word) or len(word) != len(mask):
        raise ValueError(f"Enter {len(mask)} letters for {key}.")
    if any(a != "?" and a != b for a, b in zip(mask, word)):
        raise ValueError(f"Crossing entries require {mask}. Clear or edit the conflicting entry first.")
    return {**fills, key: word}


def check(puzzle, fills):
    rows = PUZZLES[puzzle][0]
    known = letters(puzzle, fills)
    total = sum(len(row.replace("#", "")) for row in rows)
    wrong = sum(rows[y][x] != ch for (y, x), ch in known.items())
    if len(known) == total and not wrong:
        return "Solved! Every square matches."
    return f"{len(known)} of {total} squares filled · {wrong} incorrect filled squares."


def grid(puzzle, key, fills):
    rows = PUZZLES[puzzle][0]
    entries = slots(puzzle)
    numbers = {cells[0]: k[:-1] for k, (_, cells) in entries.items()}
    active = set(entries[key][1])
    ink = letters(puzzle, fills)
    tiles = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            cell = y, x
            cls = "block" if ch == "#" else "selected" if cell in active else ""
            tiles.append(f'<div class="tile {cls}"><small>{numbers.get(cell, "")}</small>'
                         f'<b>{html.escape(ink.get(cell, ""))}</b></div>')
    return f'<div class="board" style="--cols:{len(rows[0])}">' + "".join(tiles) + "</div>"


def terms(text, limit=12):
    values = list(dict.fromkeys(w.strip() for w in re.split(r"[,;\n]", text) if w.strip()))
    if len(values) > limit or any(len(w) > 64 for w in values):
        raise ValueError(f"Use up to {limit} entries, with at most 64 characters each.")
    return values


def matches(word, mask):
    mask = mask.strip().upper().replace("_", "?").replace(".", "?")
    if mask and not re.fullmatch(r"[A-Z?]{1,32}", mask):
        raise ValueError("Use letters and ? for the pattern, for example ?R??E.")
    return bool(re.fullmatch(r"[a-zA-Z]{1,32}", word) and
                (not mask or re.fullmatch(mask.replace("?", "."), word.upper())))
