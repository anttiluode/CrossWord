"""Frozen causal language models, item-slot patching, and crossword candidate scores."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crossword import matches, terms

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
          "Qwen/Qwen3-1.7B"]
LOCK = threading.RLock()
ACTIVE = None


def load_model(name):
    global ACTIVE
    name = (name or "").strip()
    if os.getenv("SPACE_ID") and name not in MODELS:
        raise ValueError("Select a preset model on this Space.")
    if not name or (not Path(name).is_dir() and not re.fullmatch(r"[\w.-]+/[\w.-]+", name)):
        raise ValueError("Choose a Hugging Face model ID or an existing local model folder.")
    with LOCK:
        if ACTIVE is None or ACTIVE.name != name:
            ACTIVE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ACTIVE = Reader(name)
        return f"{name} · {ACTIVE.device} · {len(ACTIVE.blocks)} decoder layers"


def reader(name):
    if ACTIVE is None or ACTIVE.name != name:
        raise ValueError("Press Load model for the selected checkpoint first.")
    return ACTIVE


class Reader:
    def __init__(self, name):
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=False)
        if not self.tok.is_fast:
            raise ValueError("Slot alignment needs a fast tokenizer with character offsets.")
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, use_safetensors=True, trust_remote_code=False
        ).eval().to(self.device)
        backbone = self.model.base_model
        self.blocks = next((blocks for blocks in (
            getattr(backbone, "layers", None), getattr(backbone, "h", None),
            getattr(getattr(backbone, "decoder", None), "layers", None)
        ) if blocks is not None and len(blocks)), None)
        if self.blocks is None:
            raise ValueError("This model's decoder layout is unsupported. Try a Qwen or SmolLM preset.")
        self.pad = self.tok.pad_token_id
        if self.pad is None:
            self.pad = self.tok.eos_token_id or 0
        # Support tokenizers that distinguish initial words from space-prefixed ones.
        self.choices = []
        for label in ("A", "B"):
            ids = {pieces[0] for s in (label, " " + label)
                   if len(pieces := self.tok.encode(s, add_special_tokens=False)) == 1}
            if not ids:
                raise ValueError("The tokenizer must support single-token A and B choices.")
            self.choices.append(sorted(ids))

    def encode(self, body, item=None):
        if self.tok.chat_template:
            prompt = self.tok.apply_chat_template([{"role": "user", "content": body}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = body + "\nAnswer:"
        enc = self.tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        context = getattr(self.model.config, "max_position_embeddings", 1024)
        if len(enc.input_ids) + 64 > min(1024, context):
            raise ValueError("This prompt is too long. Shorten the clues or question.")
        slot = None
        if item is not None:
            start = prompt.index(body) + len("Item: ")
            slot = next(i for i, (a, b) in enumerate(enc.offset_mapping)
                        if a <= start + len(item) - 1 < b)
        return enc.input_ids, slot

    @staticmethod
    def body(item, query):
        return f"Item: {item}\n\n{query}"

    @torch.inference_mode()
    def prototype(self, words, depth):
        """Mean of each donor's final item subtoken, at each selected layer output."""
        sums = []
        modules = self.modules(depth)
        for word in words:
            ids, pos = self.encode(self.body(word, "What kind of thing is this?"), word)
            state = []

            def capture(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                state.append(tensor[0, pos].float().clone())

            with ExitStack() as stack:
                for module in modules:
                    stack.enter_context(module.register_forward_hook(capture))
                self.model.base_model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
            sums = [a + b for a, b in zip(sums, state)] if sums else state
        return [v / len(words) for v in sums]

    def modules(self, depth):
        if not 0 <= int(depth) <= len(self.blocks):
            raise ValueError(f"Choose 0 to {len(self.blocks)} decoder blocks for this model.")
        return [self.model.get_input_embeddings(), *self.blocks[:int(depth)]]

    def patch(self, states, positions):
        stack = ExitStack()
        if states is None:
            return stack
        try:
            for module, vector in zip(self.modules(len(states)-1), states):
                def replace(module, args, output, value=vector):
                    tensor = output[0] if isinstance(output, tuple) else output
                    tensor = tensor.clone()
                    tensor[range(len(positions)), positions] = value.to(tensor.dtype)
                    return (tensor, *output[1:]) if isinstance(output, tuple) else tensor
                stack.enter_context(module.register_forward_hook(replace))
        except BaseException:
            stack.close()
            raise
        return stack

    @torch.inference_mode()
    def generate(self, body, states=None, tokens=36):
        ids, pos = self.encode(body, "thing" if states is not None else None)
        inputs = torch.tensor([ids], device=self.device)
        # Reapply patches during full forwards, including after the first generated token.
        with self.patch(states, [pos]):
            out = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                       max_new_tokens=tokens, use_cache=states is None, pad_token_id=self.pad)
        return self.tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def margins(self, bodies, states=None):
        results = []
        for i in range(0, len(bodies), 4):
            samples = [self.encode(b, "thing" if states is not None else None) for b in bodies[i:i+4]]
            lengths = [len(ids) for ids, _ in samples]
            ids = torch.full((len(samples), max(lengths)), self.pad, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, (seq, _) in enumerate(samples):
                ids[row, :len(seq)] = torch.tensor(seq, device=self.device)
                mask[row, :len(seq)] = 1
            with self.patch(states, [pos for _, pos in samples]):
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            last = logits[range(len(samples)), [n-1 for n in lengths]].float()
            yes, no = [last[:, choice].logsumexp(1) for choice in self.choices]
            results.extend((yes-no).tolist())
        return results


def proposal_words(reply, mask):
    # Accept list entries only. A sentence about the clue is not a candidate.
    pieces = re.split(r"[,;\n]", reply)
    words = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip(" .*`\t").lower() for p in pieces]
    return list(dict.fromkeys(w for w in words if matches(w, mask)))[:16]


def ask_crossing(name, left, right, mix, depth, question):
    a, b = terms(left), terms(right)
    if not a or not question.strip() or len(question) > 500:
        raise ValueError("Enter a left group and a question of 1–500 characters.")
    with LOCK:
        lm = reader(name)
        pa = lm.prototype(a, depth)
        if b:
            pb = lm.prototype(b, depth)
            pa = [torch.lerp(x, y, float(mix)) for x, y in zip(pa, pb)]
        latent = lm.generate(lm.body("thing", question), pa)
        groups = "Group A: " + ", ".join(a) + ("\nGroup B: " + ", ".join(b) if b else "")
        plain = lm.generate(groups + "\n" + question)
        return latent, plain


def rank(name, left, right, candidates, mask, depth):
    a, b = terms(left), terms(right)
    matches("test", mask)
    if not a or not b:
        raise ValueError("Enter clue words for both sides.")
    pool = terms(candidates, 24)
    with LOCK:
        lm = reader(name)
        groups = f"First group: {', '.join(a)}. Second group: {', '.join(b)}."
        if not pool:
            result = lm.generate(groups + f"\nList six single English words that fit both groups, "
                                 f"possibly in different senses. Pattern: {mask or 'any'}. Commas only.", tokens=58)
            pool = proposal_words(result, mask)
        pool = list(dict.fromkeys(w.lower() for w in pool if matches(w, mask)))
        if not pool:
            raise ValueError("No candidate matches the letter pattern. Enter your own shortlist.")
        queries = [f'Could the word "{w}" belong with the item above in one of its ordinary meanings? '
                   "A = yes, B = no. Reply with just A or B." for w in pool]
        bodies = [lm.body("thing", q) for q in queries]
        neutral = lm.margins(bodies)
        scores = []
        for group in (a, b):
            margins = lm.margins(bodies, lm.prototype(group, depth))
            scores.append([m - n for m, n in zip(margins, neutral)])
        def joint(context):
            return lm.margins([f'{context}\nCould "{w}" fit BOTH groups, possibly in different senses? '
                               'A = yes, B = no. Reply with just A or B.' for w in pool])
        plain, blank = joint(groups), joint("First group: unspecified. Second group: unspecified.")
        rows = [(w, x, y, min(x, y), p-q) for w, x, y, p, q in zip(pool, *scores, plain, blank)]
        return [[w, *[round(v, 3) for v in scores_]] for w, *scores_ in
                sorted(rows, key=lambda row: row[3], reverse=True)]


def grid_hint(name, clue, mask):
    with LOCK:
        reply = reader(name).generate(
            f"Crossword clue: {clue}\nLength: {len(mask)}. Pattern: {mask} (? is unknown). "
            "List three possible single-word answers, separated by commas. Only words.", tokens=42)
    found = proposal_words(reply, mask)
    return ("Pattern matches: " + ", ".join(w.upper() for w in found) if found else
            "No proposed word matched the crossing letters.") + "\n\nModel output: " + reply


def solve_clue(name, clue, mask, candidates):
    if not clue.strip() or len(clue) > 500:
        raise ValueError("Enter a crossword clue of 1–500 characters.")
    matches("test", mask)
    supplied = terms(candidates, 24)
    with LOCK:
        lm = reader(name)
        raw = ""
        if not supplied:
            raw = lm.generate(f"Crossword clue: {clue}\nLetter pattern: {mask or 'any'} "
                              "(? means one unknown letter). Give up to six candidate answers. "
                              "Single words only, separated by commas.", tokens=58)
            supplied = proposal_words(raw, mask)
        pool = list(dict.fromkeys(w.lower() for w in supplied if matches(w, mask)))
        if not pool:
            return [], raw or "No candidate matches the letter pattern."
        def judge(text):
            return lm.margins([f'Crossword clue: {text}\nCandidate solution: {w}\n'
                               'Does this solve the clue? A = yes, B = no. Reply with just A or B.'
                               for w in pool])
        values, neutral = judge(clue), judge("(unspecified)")
        scored = sorted(zip(pool, (v-n for v, n in zip(values, neutral))), key=lambda pair: -pair[1])
        return [[w, round(s, 3)] for w, s in scored], raw or "Ranked the supplied shortlist."
