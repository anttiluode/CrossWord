"""Local Gradio app and Hugging Face Space entry point."""
import os

import gradio as gr
import torch

import engine
from crossword import terms

CSS = """
.gradio-container {max-width:900px!important; margin:auto}
.mast {border-bottom:2px solid #20745d; padding:14px 0; margin-bottom:8px}
.mast h1 {font:bold clamp(2rem,5vw,3.2rem) Georgia,serif; letter-spacing:-.04em; margin:0}
"""


def solve(model, clue, mask, shortlist, related):
    try:
        rows, notes = engine.solve_clue(model, clue, mask, shortlist, related)
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    headers = ["Answer", "Clue score"]
    if terms(related):
        headers.append("Second meaning")
    return gr.update(headers=headers, datatype=["str"] + ["number"] * (len(headers)-1),
                     value={"headers": headers, "data": rows}), notes


def build():
    with gr.Blocks(title="CrossWord Solver", css=CSS, theme=gr.themes.Soft(primary_hue="emerald")) as ui:
        gr.HTML('<header class="mast"><h1>CrossWord Solver</h1></header>')
        clue = gr.Textbox(label="Crossword clue", placeholder="A bird or a construction machine",
                          autofocus=True)
        with gr.Row():
            mask = gr.Textbox(label="Known letters", placeholder="?R??E",
                              info="One ? per unknown letter. Leave blank for any length.")
            related = gr.Textbox(value="", label="Second meaning (optional)", placeholder="hoist, winch, pulley",
                                 info="Related words for another sense of the answer.")
        with gr.Accordion("Candidate words (optional)", open=False):
            shortlist = gr.Textbox(label="Words to compare", placeholder="crane, brake, grape",
                                   info="Separate candidates with commas. Leave empty for model suggestions.")
        with gr.Row():
            model = gr.Dropdown(engine.MODELS, value=engine.MODELS[-1] if torch.cuda.is_available()
                                else engine.MODELS[1], label="Model", scale=2,
                                allow_custom_value=not bool(os.getenv("SPACE_ID")),
                                info="Loads on the first solve.")
            go = gr.Button("Solve clue", variant="primary", scale=1)
        table = gr.Dataframe(headers=["Answer", "Clue score"], datatype=["str", "number"],
                             interactive=False, label="Ranked answers")
        with gr.Accordion("Proposal details", open=False):
            details = gr.Textbox(label="Candidate sources", lines=2, interactive=False)
        inputs = [model, clue, mask, shortlist, related]
        go.click(solve, inputs, [table, details], api_name="solve_clue")
        clue.submit(solve, inputs, [table, details], api_name=False)
    return ui


if __name__ == "__main__":
    build().queue(max_size=16).launch()
