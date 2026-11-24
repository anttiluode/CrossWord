"""Frozen causal language models, item-slot patching, and crossword candidate scores."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crossword import matches, normalize_mask, terms

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
        raise ValueError("The selected checkpoint is not loaded.")
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
    def generate(self, body, tokens=58):
        ids, _ = self.encode(body)
        inputs = torch.tensor([ids], device=self.device)
        out = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                  max_new_tokens=tokens, pad_token_id=self.pad)
        return self.tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def token_proposals(self, prompt, mask):
        ids, _ = self.encode(prompt)
        inputs = torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=inputs, use_cache=False).logits[0, -1]
        ids = logits.topk(min(8192, len(self.tok), logits.shape[0])).indices.tolist()
        special = set(self.tok.all_special_ids)
        size = len(self.tok)
        ids = [i for i in ids if i < size and i not in special]
        words = self.tok.batch_decode([[i] for i in ids], clean_up_tokenization_spaces=False)
        found = {}
        for word in words:
            word = word.strip().lower()
            if matches(word, mask):
                found[word] = None
                if len(found) == 8:
                    break
        return list(found)

    @torch.inference_mode()
    def phrase_proposals(self, context, mask):
        """Back up free generation with a small beam of word-boundary token completions."""
        prompt, _ = self.encode(context + "\nReply with only the matching phrase.")
        # Decode once for this fallback. Continuations must actually start a new word.
        if not hasattr(self, "_word_tokens"):
            special = set(self.tok.all_special_ids)
            ids = [i for i in self.tok.get_vocab().values() if i not in special]
            pieces = self.tok.batch_decode([[i] for i in ids], clean_up_tokenization_spaces=False)
            self._word_tokens = [(i, piece) for i, piece in zip(ids, pieces)
                                 if re.fullmatch(r" ?[a-zA-Z]{1,32}", piece)]
        beam = [(0.0, [], [])]
        for index, part in enumerate(mask.split()):
            pattern = re.compile(part.replace("?", "."), re.IGNORECASE)
            allowed = [(i, token.strip().lower()) for i, token in self._word_tokens
                       if (index == 0 or token.startswith(" ")) and pattern.fullmatch(token.strip())]
            if not allowed:
                return {}
            sequences = [prompt + suffix for _, suffix, _ in beam]
            inputs = torch.tensor(sequences, device=self.device)
            logits = self.model(input_ids=inputs, use_cache=False).logits[:, -1].float().log_softmax(-1)
            allowed = [(i, word) for i, word in allowed if i < logits.shape[-1]]
            if not allowed:
                return {}
            indices, words = zip(*allowed)
            scores = logits[:, list(indices)].tolist()
            options = [(total + score, suffix + [i], prefix + [word])
                       for (total, suffix, prefix), row in zip(beam, scores)
                       for i, word, score in zip(indices, words, row)]
            unique = {}
            for item in sorted(options, key=lambda o: -o[0]):
                unique.setdefault(tuple(item[2]), item)
                if len(unique) == 4:
                    break
            beam = list(unique.values())
        best = beam[0][0]
        return {" ".join(words): (score - best) / len(words) for score, _, words in beam}

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


def solve_clue(name, clue, mask, candidates, related=""):
    clue = clue.strip()
    if len(clue) > 500:
        raise ValueError("Keep the crossword clue under 501 characters.")
    mask = normalize_mask(mask)
    supplied = terms(candidates, 24)
    group = terms(related)
    if not clue and group:
        clue, group = ", ".join(group), []
    if not clue and not mask:
        raise ValueError("Enter a clue, known letters, or a meaning hint.")
    with LOCK:
        load_model(name)
        lm = reader(name.strip())
        phrase_scores = {}
        if supplied:
            proposed = supplied
            details = "Ranked your shortlist."
        else:
            context = f"Crossword clue: {clue}" if clue else "Fill this crossword pattern."
            if mask:
                squares = len(mask.replace(" ", ""))
                context += f"\n{squares} letters. Pattern: {mask} (? = unknown)."
            if group:
                context += "\nAnother sense is related to: " + ", ".join(group) + "."
            form = "answers" if " " in mask or not mask else "single-word answers"
            raw = lm.generate(context + f"\nGive six possible {form}, separated by commas.")
            proposed = proposal_words(raw, mask)
            # Free generation also covers answers that need several tokenizer pieces.
            # A second read supplies literal matching spellings from likely first tokens.
            extras = [] if " " in mask else lm.token_proposals(context + "\nReply with one answer word.", mask)
            proposed.extend(extras)
            details = f"Model proposals: {raw}"
            if extras:
                details += "\nSingle-token suggestions: " + ", ".join(extras)
            if not proposed and " " in mask:
                phrase_scores = lm.phrase_proposals(context, mask)
                proposed = list(phrase_scores)
                details += "\nPattern-guided phrase suggestions: " + (", ".join(proposed) or "none")
        pool = list(dict.fromkeys(" ".join(w.lower().split()) for w in proposed if matches(w, mask)))
        if not pool:
            return [], "No answers matched. Check the letters or enter a shortlist.\n" + details

        if not clue:
            return [[w] for w in pool], details + "\nLetter matches in proposal order."

        def judge(text):
            return lm.margins([f'Crossword clue: {text}\nCandidate solution: {w}\n'
                               'Does this solve the clue? A = yes, B = no. Reply with just A or B.'
                               for w in pool])

        direct = [v - b + phrase_scores.get(word, 0)
                  for word, v, b in zip(pool, judge(clue), judge("(unspecified)"))]
        order = sorted(range(len(pool)), key=lambda i: -direct[i])
        if not group:
            return [[pool[i], round(direct[i], 3)] for i in order], details

        bodies = [lm.body("thing", f'Could "{w}" belong with the item above in one of its ordinary '
                          'meanings? A = yes, B = no. Reply with just A or B.') for w in pool]
        base = lm.margins(bodies)
        patched = lm.margins(bodies, lm.prototype(group, min(4, len(lm.blocks))))
        second = [v-b for v, b in zip(patched, base)]
        # Rank each signal separately. Ties share a rank; scales can differ by prompt.
        def positions(values):
            return [sum(other > value for other in values) for value in values]

        first_rank, second_rank = positions(direct), positions(second)
        order.sort(key=lambda i: (max(first_rank[i], second_rank[i]), first_rank[i] + second_rank[i]))
        return [[pool[i], round(direct[i], 3), round(second[i], 3)] for i in order], details
