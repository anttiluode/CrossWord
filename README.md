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

Enter a crossword clue from your own puzzle, add any known letters, and press **Solve clue**. The selected model loads automatically. Answers are English single-word fills of up to 32 letters.

- For **A bird or a construction machine**, try `?R??E`. A question mark is one missing letter. Five question marks mean any five-letter answer.
- For **Marsh bird**, add `hoist, winch, pulley` under **Second meaning** to favor answers that also fit the equipment sense.
- **Candidate words** lets you rank your own shortlist. With an empty shortlist, the model proposes answers and the solver adds matching spellings from likely next tokens. The pattern filters every candidate before scoring.
- **Proposal details** shows where the suggestions came from. A single-token suggestion can be a fragment. The solver can miss valid answers.

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

On Linux or macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` for the environment. Open the local URL printed by Gradio.

For NVIDIA acceleration, install a CUDA-enabled PyTorch build matching your driver into the same environment. Check with `python -c "import torch; print(torch.cuda.is_available())"`. CPU inference uses float32. CUDA uses bfloat16 when supported and float16 otherwise. The first model load downloads weights. For offline use, select a complete local Transformers model folder or set `HF_HUB_OFFLINE=1` after caching a model. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

## Models and Spaces

Locally, select a preset or type a compatible Hugging Face ID or model folder. Models need safetensors weights, a fast tokenizer with offsets, and a supported causal decoder layout, such as Qwen or Llama. GGUF files and Ollama endpoints do not expose the hidden layers this app edits.

To host it, create a **Gradio Space** and copy the repository files to its root, including this README and `requirements.txt`. The YAML header supplies the entry point and SDK version. The Space starts without loading weights. Visitors share one active checkpoint and can select among the three presets. Each solve uses the selected checkpoint. Switching to a different checkpoint releases the previous one.

## How scoring works

The clue score measures how much the clue changes the model's yes/no log odds for each answer relative to an unspecified clue. Higher scores rank first.

With a second meaning, the solver averages the supplied related words at the item embedding and the first four decoder blocks (or fewer on shallower models). It patches those vectors into a neutral item slot and measures how much they change the candidate's membership score. Each candidate receives a rank for each meaning. The solver favors the best worst rank, breaking ties by rank sum and then by the original clue ranking. The two displayed scores measure different prompts and have different scales.

For open-ended exploration of blended word states, see [Category](https://github.com/anttiluode/Category).

## Verify

Run `python -m unittest -v` for the letter-pattern checks. Set `CROSSWORD_TEST_MODEL` to a cached ID or local checkpoint to exercise both solver paths and the single-token activation transplant.
