"""Input and letter constraints for the crossword clue solver."""
import re


def terms(text, limit=12):
    values = list(dict.fromkeys(w.strip() for w in re.split(r"[,;\n]", text) if w.strip()))
    if len(values) > limit or any(len(w) > 64 for w in values):
        raise ValueError(f"Use up to {limit} entries, with at most 64 characters each.")
    return values


def normalize_mask(mask):
    mask = mask.strip().upper().replace("_", "?").replace(".", "?")
    if mask and not re.fullmatch(r"[A-Z?]{1,32}", mask):
        raise ValueError("Use up to 32 letters and ? for unknown squares, for example ?R??E.")
    return mask


def matches(word, mask):
    mask = normalize_mask(mask)
    return bool(re.fullmatch(r"[a-zA-Z]{1,32}", word) and
                (not mask or re.fullmatch(mask.replace("?", "."), word.upper())))
