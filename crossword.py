"""Input and letter constraints for the crossword clue solver."""
import re


def terms(text, limit=12):
    values = list(dict.fromkeys(w.strip() for w in re.split(r"[,;\n]", text) if w.strip()))
    if len(values) > limit or any(len(w) > 64 for w in values):
        raise ValueError(f"Use up to {limit} entries, with at most 64 characters each.")
    return values


def normalize_mask(mask):
    mask = " ".join(mask.upper().split()).replace("_", "?").replace(".", "?")
    if len(mask.replace(" ", "")) > 32 or (mask and not re.fullmatch(r"[A-Z?]+(?: [A-Z?]+)*", mask)):
        raise ValueError("Use up to 32 letters or ? squares. Separate words with spaces, as in N?W Y?RK.")
    return mask


def matches(word, mask):
    mask = normalize_mask(mask)
    word = " ".join(word.split())
    return bool(len(word.replace(" ", "")) <= 32 and re.fullmatch(r"[a-zA-Z]+(?: [a-zA-Z]+)*", word) and
                (not mask or re.fullmatch(mask.replace("?", "."), word.upper())))
