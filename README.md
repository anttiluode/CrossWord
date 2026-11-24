---
title: CrossWord
emoji: 🧩
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# CrossWord

Play a crossword, ask a frozen language model for an answer that fits your letters, or give a missing word two different meanings and watch the candidates compete.

## Try a semantic crossing

1. Load a checkpoint in **Model desk**. SmolLM2 is the smallest download. Qwen2.5 is a good next comparison. Qwen3-1.7B is the checkpoint used in the related Rakeisuus experiments.
2. Open **Semantic crossings**. Start with *heron, stork, egret* on one side and *hoist, winch, pulley* on the other. **Rank crossing words** scores *crane* alongside five alternatives. Edit either group, change the shortlist, or enter a pattern such as `?R??E`.
3. **Ask the blended state** accepts your own question. Try “Where might I see this?” at mix 0, 0.5 and 1. Read the answer from the patched item beside the answer obtained by printing both groups in the prompt. The text baseline sees the same groups at each slider position. For a single group's common thread, leave B empty.
4. The two groups stay separate for ranking. The blend slider changes the free-question experiment. Leaving the shortlist empty asks the model for proposals, so a missing answer may reflect proposal quality.

In a CPU run of the built-in six-candidate marsh/construction crossing, Qwen2.5-0.5B-Instruct ranked **winch** first and **crane** second using the weaker of the two activation scores. The plain-prompt column ranked **crane** first. This is one puzzle, one prompt and one patch depth. Change a clue or switch models and compare.

## Play or solve your own clue

- **Crossword grid** has three original puzzles. Pick a numbered Across or Down clue, enter an answer and press Enter. Shaded squares mark the selected entry. Writing a word fills its shared letters immediately. Clearing an entry preserves letters supplied by other entries. **Check squares** checks all visible letters.
- **Ask model for candidates** sends the active clue, length and known crossing letters. The hint code has no access to the solution grid. The raw response is visible even if the model misses the pattern.
- **Clue desk** accepts a clue from any puzzle. Enter your known letters, such as `?R??E`, and optionally a shortlist. It filters candidate spellings and measures how much the clue improves the model's score for each. Use English single-word fills with A–Z letters. A question mark stands for exactly one letter.

## Run locally

Python 3.11 or 3.12. On Windows Command Prompt:

```bat
git clone https://github.com/anttiluode/CrossWord.git
cd CrossWord
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

On Linux or macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` for the environment. Open the local URL printed by Gradio. Grid play works before downloading any weights.

For NVIDIA acceleration, install a CUDA-enabled PyTorch build matching your driver into the same environment. Check with `python -c "import torch; print(torch.cuda.is_available())"`. The model desk reports the actual device. CPU inference uses float32. CUDA uses bfloat16 when supported and float16 otherwise. The first model load downloads weights. For offline use, select a complete local Transformers model folder or set `HF_HUB_OFFLINE=1` after caching a model. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

## Models and Spaces

Locally, select a preset or type a compatible Hugging Face ID or model folder. Models need safetensors weights, a fast tokenizer with offsets, and a supported causal decoder layout, such as Qwen or Llama. GGUF files and Ollama endpoints do not expose the hidden layers this app edits.

To host it, create a **Gradio Space** and copy the repository files to its root, including this README and `requirements.txt`. The YAML header supplies the entry point and SDK version. The Space starts without loading weights. Visitors share one active checkpoint and can select among the three presets. Each browser session has its own crossword fills. Loading a new checkpoint releases the previous one; select and load your preferred model again if another session switches it.

## How scoring works

Each clue word or phrase gets its own forward pass in a common `Item: …` template. The app captures the last subtoken of the item at the embedding and after each selected early decoder block. It averages donor vectors **separately at each depth**. The neutral recipient contains `Item: thing`. Hooks replace that slot at the same selected depths, leaving the question and the checkpoint weights intact. Multi-token phrases use only their final subtoken and can behave differently from single-token words.

For each candidate, the model judges membership as A (yes) or B (no). Group A and Group B columns show the change in A-versus-B log odds relative to the unpatched neutral item. The ranking uses the smaller group score so that a strong fit to one group cannot compensate for a weak fit to the other. The plain-prompt column uses both printed lists and subtracts the score with both groups unspecified. Compare rankings within each method. Scores are sensitive to wording, tokenizer, checkpoint and layer depth.

Free questions interpolate the two group means at each selected depth. The patched version decodes greedily, reapplying the intervention through generation. Neither a fluent answer nor a high rank establishes what information a model ordinarily uses.

## Verify

`python -m unittest -v` exercises the grids and crossing constraints. Set `CROSSWORD_TEST_MODEL` to a cached ID or local checkpoint to also compare a single-token transplant against a direct-word prompt and check that removing the hooks restores the original score.
