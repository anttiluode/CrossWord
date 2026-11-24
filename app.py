"""Run with python app.py or use this file as the Gradio Space entry point."""
import os
from functools import wraps

import gradio as gr

import crossword as cw
import engine

CSS = """
.gradio-container {max-width:1120px!important; margin:auto; --block-radius:10px}
.mast {border-bottom:2px solid #20745d; padding:20px 0 14px; margin-bottom:10px}
.mast h1 {font: bold clamp(2rem,5vw,3.6rem) Georgia,serif; letter-spacing:-.055em; margin:0}
.mast p {margin:.3rem 0 0}
.board {display:grid; grid-template-columns:repeat(var(--cols),1fr); aspect-ratio:1;
 width:min(100%,340px); margin:8px auto 16px; border:2px solid #253b35; background:#253b35; gap:1px}
.tile {position:relative; display:grid; place-items:center; background:#fffaf0; color:#153c32}
.tile.selected {background:#bee8cd; box-shadow:inset 0 -3px #28745b}
.tile.block {background:#253b35}
.tile small {position:absolute; left:5px; top:2px; font:600 .8rem system-ui}
.tile b {font:600 clamp(1.2rem,4vw,2rem) Georgia,serif}
"""
EXAMPLES = {
    "Marsh / construction": ("heron, stork, egret", "hoist, winch, pulley", "?????",
                              "crane, heron, winch, stork, cable, egret"),
    "Pond / concert": ("trout, perch, salmon", "guitar, cello, drums", "????",
                       "bass, carp, drum, lute, pike"),
    "Tree / dog": ("trunk, wood, rind", "growl, howl, yelp", "????", "bark, howl, root, limb"),
}


def friendly(fn):
    @wraps(fn)
    def wrapped(*args):
        try:
            return fn(*args)
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def labels(puzzle):
    return [(f"{key[:-1]} {'Across' if key.endswith('A') else 'Down'} · {clue} ({len(cells)})", key)
            for key, (clue, cells) in sorted(cw.slots(puzzle).items(),
                key=lambda pair: (pair[0][-1], int(pair[0][:-1])))]


def view(puzzle, key, fills):
    clue = cw.slots(puzzle)[key][0]
    mask = cw.pattern(puzzle, key, fills)
    return cw.grid(puzzle, key, fills), f"**{key}** · {clue} · `{mask}`", fills.get(key, ""), ""


def switch(puzzle):
    options = labels(puzzle)
    key = options[0][1]
    return {}, gr.Dropdown(choices=options, value=key), *view(puzzle, key, {}), ""


def update(puzzle, key, fills, word=None):
    if word is None:
        fills = {k: v for k, v in fills.items() if k != key}
    else:
        fills = cw.enter(puzzle, key, fills, word)
    return fills, *view(puzzle, key, fills), ""


def build():
    puzzle = next(iter(cw.PUZZLES))
    key = labels(puzzle)[0][1]
    initial = view(puzzle, key, {})
    with gr.Blocks(title="CrossWord", css=CSS, theme=gr.themes.Soft(primary_hue="emerald")) as ui:
        gr.HTML('<header class="mast"><h1>CrossWord</h1><p>Crossing letters and crossing meanings</p></header>')
        with gr.Accordion("Model desk · load a checkpoint for hints and experiments", open=True):
            with gr.Row():
                model = gr.Dropdown(engine.MODELS, value=engine.MODELS[0], label="Checkpoint",
                                    allow_custom_value=not bool(os.getenv("SPACE_ID")), scale=3)
                load = gr.Button("Load model", scale=1)
            status = gr.Textbox(label="Model status", value="Grid ready · model optional", interactive=False)
        with gr.Tab("Crossword grid"):
            fills = gr.State({})
            with gr.Row():
                with gr.Column(scale=4):
                    edition = gr.Dropdown(list(cw.PUZZLES), value=puzzle, label="Puzzle")
                    board = gr.HTML(initial[0])
                    check = gr.Button("Check squares")
                    verdict = gr.Textbox(label="Check result", interactive=False)
                with gr.Column(scale=6):
                    entry = gr.Dropdown(labels(puzzle), value=key, label="Across and Down")
                    active = gr.Markdown(initial[1])
                    answer = gr.Textbox(label="Your answer", placeholder="Type a word and press Enter")
                    with gr.Row():
                        write = gr.Button("Write answer", variant="primary")
                        clear = gr.Button("Clear entry")
                    hint = gr.Button("Ask model for candidates")
                    suggestions = gr.Textbox(label="Model hint", lines=3, interactive=False)
        with gr.Tab("Clue desk"):
            free_clue = gr.Textbox(label="Crossword clue", value="A bird or a construction machine")
            with gr.Row():
                free_mask = gr.Textbox(label="Crossing letters", value="?R??E",
                                      info="Use ? for unknown squares, or leave empty.")
                free_candidates = gr.Textbox(label="Your candidate words", scale=2,
                                      info="Separate with commas. Leave empty for model proposals.")
            solve = gr.Button("Solve clue", variant="primary")
            free_table = gr.Dataframe(headers=["Candidate", "Clue evidence"], datatype=["str", "number"],
                                      interactive=False, label="Candidates matching the letters")
            free_output = gr.Textbox(label="Proposal details", interactive=False)
        with gr.Tab("Semantic crossings"):
            gr.Markdown("Give the missing word two clues. Each group gets its own hidden-state average.")
            example = gr.Dropdown(list(EXAMPLES), value=next(iter(EXAMPLES)), label="Example")
            first = next(iter(EXAMPLES.values()))
            with gr.Row():
                left = gr.Textbox(value=first[0], label="Clue group A", info="Separate words or phrases with commas.")
                right = gr.Textbox(value=first[1], label="Clue group B")
            with gr.Row():
                mask = gr.Textbox(value=first[2], label="Letter pattern", info="? = unknown. Leave blank for any length.")
                candidates = gr.Textbox(value=first[3], label="Candidate words", scale=3,
                    info="Edit the shortlist or leave it empty to have the model propose words.")
            depth = gr.Slider(0, 12, value=4, step=1, label="Early decoder blocks to patch",
                              info="0 patches only the embedding. Depth applies to both experiments.")
            rank = gr.Button("Rank crossing words", variant="primary")
            table = gr.Dataframe(headers=["Word", "Group A", "Group B", "Weaker fit ↓", "Plain prompt"],
                                 datatype=["str", "number", "number", "number", "number"],
                                 interactive=False, label="Change in yes/no log odds over the neutral prompt")
            gr.Markdown("### Ask the blended state")
            with gr.Row():
                mix = gr.Slider(0, 1, value=0.5, step=.05, label="Mix · 0 is A, 1 is B")
                question = gr.Textbox(value="What kind of thing is the item above? Answer briefly.",
                                      label="Your question", scale=3)
            ask = gr.Button("Ask model")
            with gr.Row():
                latent = gr.Textbox(label="Patched item", interactive=False, lines=3)
                plain = gr.Textbox(label="Same question with the clue words in text", interactive=False, lines=3)
        model.input(lambda: "Press Load model to use this selection.", outputs=status)
        load.click(friendly(engine.load_model), model, status)
        readouts = [board, active, answer, suggestions]
        edition.input(switch, edition, [fills, entry, *readouts, verdict])
        entry.input(view, [edition, entry, fills], readouts)
        for event in [write.click, answer.submit]:
            event(friendly(update), [edition, entry, fills, answer], [fills, *readouts, verdict])
        clear.click(update, [edition, entry, fills], [fills, *readouts, verdict])
        check.click(cw.check, [edition, fills], verdict)
        hint.click(friendly(lambda n, p, k, f: engine.grid_hint(n, cw.slots(p)[k][0], cw.pattern(p,k,f))),
                   [model, edition, entry, fills], suggestions)
        example.input(lambda name: (*EXAMPLES[name], [], "", ""), example,
                      [left, right, mask, candidates, table, latent, plain])
        solve.click(friendly(engine.solve_clue), [model, free_clue, free_mask, free_candidates],
                    [free_table, free_output])
        rank.click(friendly(engine.rank), [model, left, right, candidates, mask, depth], table)
        ask.click(friendly(engine.ask_crossing), [model, left, right, mix, depth, question], [latent, plain])
    return ui


if __name__ == "__main__":
    build().queue(max_size=16).launch()
