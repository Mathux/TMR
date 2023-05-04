from functools import partial
import os

import torch
import numpy as np
import gradio as gr
import gdown

from load import load_model, load_json
from load import load_unit_motion_embs_splits, load_keyids_splits


WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis </h1>

<h2 style='text-align: center'>
<a href="https://mathis.petrovich.fr" target="_blank"><nobr>Mathis Petrovich</nobr></a> &emsp;
<a href="https://ps.is.mpg.de/~black" target="_blank"><nobr>Michael J. Black</nobr></a> &emsp;
<a href="https://imagine.enpc.fr/~varolg" target="_blank"><nobr>G&uumll Varol</nobr></a>
</h2>

<h2 style='text-align: center'>
<nobr>arXiv 2023</nobr>
</h2>

<h3 style="text-align:center;">
<a target="_blank" href="https://arxiv.org/abs/XXXX.XXXXX"> <button type="button" class="btn btn-primary btn-lg"> Paper </button></a>
<a target="_blank" href="https://github.com/Mathux/TMR"> <button type="button" class="btn btn-primary btn-lg"> Code </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/tmr"> <button type="button" class="btn btn-primary btn-lg"> Webpage </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/tmr/tmr.bib"> <button type="button" class="btn btn-primary btn-lg"> BibTex </button></a>
</h3>

<h3> Description </h3>
<p>
This space illustrates <a href='https://mathis.petrovich.fr/tmr/' target='_blank'><b>TMR</b></a>, a method for text-to-motion retrieval. Given a gallery of 3D human motions (which can be unseen during training) and a text query, the goal is to search for motions which are close to the text query.
</p>
</div>
"""

EXAMPLES = [
    "A person is walking slowly",
    "A person is walking in a circle",
    "A person is jumping rope",
    "Someone is doing a backflip",
    "A person is doing a moonwalk",
    "A person walks forward and then turns back",
    "Picking up an object",
    "A person is swimming in the sea",
    "A human is squatting",
    "Someone is jumping with one foot",
    "A person is chopping vegetables",
    "Someone walks backward",
    "Somebody is ascending a staircase",
    "A person is sitting down",
    "A person is taking the stairs",
    "Someone is doing jumping jacks",
    "The person walked forward and is picking up his toolbox",
    "The person angrily punching the air"
]

# Show closest text in the training


# css to make videos look nice
# var(--block-border-color);
CSS = """
.retrieved_video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}

.contour_video {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: var(--layer-5);
    border-radius: var(--block-radius);
    background: var(--background-fill-primary);
    padding: 0 var(--size-6);
    max-height: var(--size-screen-h);
    overflow: hidden;
}
"""


DEFAULT_TEXT = "A person is "

def humanml3d_keyid_to_babel_rendered_url(h3d_index, amass_to_babel, keyid):
    # Don't show the mirrored version of HumanMl3D
    if "M" in keyid:
        return None

    dico = h3d_index[keyid]
    path = dico["path"]

    # HumanAct12 motions are not rendered online
    # so we skip them for now
    if "humanact12" in path:
        return None

    # This motion is not rendered in BABEL
    # so we skip them for now
    if path not in amass_to_babel:
        return None

    babel_id = amass_to_babel[path].zfill(6)
    url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"

    # For the demo, we retrieve from the first annotation only
    ann = dico["annotations"][0]
    start = ann["start"]
    end = ann["end"]
    text = ann["text"]

    data = {
        "url": url,
        "start": start,
        "end": end,
        "text": text,
        "keyid": keyid,
        "babel_id": babel_id,
        "path": path
    }

    return data


def retrieve(model, keyid_to_url, all_unit_motion_embs, all_keyids, text, splits=["test"], nmax=8):
    unit_motion_embs = torch.cat([all_unit_motion_embs[s] for s in splits])
    keyids = np.concatenate([all_keyids[s] for s in splits])

    scores = model.compute_scores(text, unit_embs=unit_motion_embs)

    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break

        data = keyid_to_url(keyid)
        if data is None:
            continue
        data["score"] = round(float(score), 2)
        datas.append(data)
    return datas


# HTML component
def get_video_html(data, video_id, width=700, height=700):
    url = data["url"]
    start = data["start"]
    end = data["end"]
    score = data["score"]
    text = data["text"]
    keyid = data["keyid"]
    babel_id = data["babel_id"]
    path = data["path"]

    trim = f"#t={start},{end}"
    title = f'''Score = {score}

Corresponding text: {text}

HumanML3D keyid: {keyid}

BABEL keyid: {babel_id}

AMASS path: {path}'''

    # class="wrap default svelte-gjihhp hide"
    # <div class="contour_video" style="position: absolute; padding: 10px;">
    # width="{width}" height="{height}"
    video_html = f'''
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}" title="{title}">
  <source src="{url}{trim}" type="video/mp4">
  Your browser does not support the video tag.
</video>
'''
    return video_html


def retrieve_component(retrieve_function, text, splits_choice, nvids, n_component=24):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(n_component)]

    # cannot produce more than n_compoenent
    nvids = min(nvids, n_component)

    if "Unseen" in splits_choice:
        splits = ["test"]
    else:
        splits = ["train", "val", "test"]

    datas = retrieve_function(text, splits=splits, nmax=nvids)
    htmls = [get_video_html(data, idx) for idx, data in enumerate(datas)]
    # get n_component exactly if asked less
    # pad with dummy blocks
    htmls = htmls + [None for _ in range(max(0, n_component-nvids))]
    return htmls


if not os.path.exists("data"):
    gdown.download_folder("https://drive.google.com/drive/folders/1MgPFgHZ28AMd01M1tJ7YW_1-ut3-4j08",
                          use_cookies=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOADING
model = load_model(device)
splits = ["train", "val", "test"]
all_unit_motion_embs = load_unit_motion_embs_splits(splits, device)
all_keyids = load_keyids_splits(splits)

h3d_index = load_json("amass-annotations/humanml3d.json")
amass_to_babel = load_json("amass-annotations/amass_to_babel.json")

keyid_to_url = partial(humanml3d_keyid_to_babel_rendered_url, h3d_index, amass_to_babel)
retrieve_function = partial(retrieve, model, keyid_to_url, all_unit_motion_embs, all_keyids)

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
retrieve_and_show = partial(retrieve_component, retrieve_function)

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(scale=2):
                text = gr.Textbox(placeholder="Type the motion you want to search with a sentence",
                                  show_label=True, label="Text prompt", value=DEFAULT_TEXT)
            with gr.Column(scale=1):
                btn = gr.Button("Retrieve", variant='primary')
                clear = gr.Button("Clear", variant='secondary')

            with gr.Row():
                with gr.Column(scale=1):
                    splits_choice = gr.Radio(["All motions", "Unseen motions"], label="Gallery of motion",
                                             value="All motions",
                                             info="The motion gallery is coming from HumanML3D")

                with gr.Column(scale=1):
                    # nvideo_slider = gr.Slider(minimum=4, maximum=24, step=4, value=8, label="Number of videos")
                    nvideo_slider = gr.Radio([4, 8, 12, 16, 24], label="Videos",
                                             value=8,
                                             info="Number of videos to display")

        with gr.Column(scale=2):
            def retrieve_example(text, splits_choice, nvideo_slider):
                return retrieve_and_show(text, splits_choice, nvideo_slider)

            examples = gr.Examples(examples=[[x, None, None] for x in EXAMPLES],
                                   inputs=[text, splits_choice, nvideo_slider],
                                   examples_per_page=20,
                                   run_on_click=False, cache_examples=False,
                                   fn=retrieve_example, outputs=[])

    i = -1
    # should indent
    for _ in range(6):
        with gr.Row():
            for _ in range(4):
                i += 1
                video = gr.HTML()
                videos.append(video)

    # connect the examples to the output
    # a bit hacky
    examples.outputs = videos

    def load_example(example_id):
        processed_example = examples.non_none_processed_examples[example_id]
        return gr.utils.resolve_singleton(processed_example)

    examples.dataset.click(
        load_example,
        inputs=[examples.dataset],
        outputs=examples.inputs_with_examples,  # type: ignore
        show_progress=False,
        postprocess=False,
        queue=False,
        ).then(
            fn=retrieve_example,
            inputs=examples.inputs,
            outputs=videos
        )

    btn.click(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    text.submit(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    splits_choice.change(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    nvideo_slider.change(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)

    def clear_videos():
        return [None for x in range(24)] + [DEFAULT_TEXT]

    clear.click(fn=clear_videos, outputs=videos + [text])

demo.launch()
