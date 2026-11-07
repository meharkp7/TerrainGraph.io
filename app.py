# app.py — HuggingFace Space entrypoint (Gradio).
# Must stay in sync with the GitHub repo: Spaces deploy from this file.
#
# Two faces:
#   Demo tab — human upload → segmentation, path, briefing, chart.
#   API tab  — machine endpoint (/segment) used by the lean Railway backend:
#              image in → mask PNG + confidence PNG + class distribution out.
import gradio as gr
from pipeline_hf import run_pipeline
from segmentor_space import encode_mask_files
from utils import create_bar_chart


def predict(image):
    if image is None:
        return None, None, None, "Please upload an image first.", None
    result = run_pipeline(image)
    chart = create_bar_chart(result["class_dist"])
    stats = (
        f"Traversability: {result['avg_trav']:.2f} · "
        f"Path: {result['path_hops']} hops"
    )
    return (
        result["original"],
        result["segmented"],
        result["overlay"],
        f"{result['briefing']}\n\n{stats}",
        chart,
    )


def api_segment(image):
    """Machine endpoint for the Railway backend (see segmentor_space.py)."""
    if image is None:
        raise gr.Error("No image provided")
    result = run_pipeline(image)
    mask_path, conf_path = encode_mask_files(
        result["mask"], result["confidence"])
    return mask_path, conf_path, result["class_dist"]


with gr.Blocks(theme=gr.themes.Soft(), title="TerrainGraph AI") as demo:
    gr.Markdown("""
    # TerrainGraph AI
    ### Intelligent Terrain Analysis & Safe Navigation
    DeepLabV3+ segmentation → Dijkstra safe-path planning → AI briefing.
    Full system (TigerGraph memory + LLM reasoning + React dashboard) on
    [GitHub](https://github.com/meharkp7/TerrainGraph.io).
    """)

    with gr.Tab("Demo"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")

        run_btn = gr.Button("Analyze Terrain", variant="primary")

        with gr.Row():
            original = gr.Image(label="Original")
            segmented = gr.Image(label="Segmentation")
            overlay = gr.Image(label="Safe Path")

        briefing = gr.Textbox(label="AI Briefing", lines=5)
        chart = gr.Plot(label="Terrain Distribution")

        run_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[original, segmented, overlay, briefing, chart],
        )

    with gr.Tab("API"):
        gr.Markdown("""
        Machine endpoint for the Railway backend.
        `POST /segment`: image in → `mask.png` (class ids 0-5) +
        `conf.png` (confidence 0-255) + class distribution JSON.
        """)
        with gr.Row():
            api_input = gr.Image(type="pil", label="Input image")

        api_btn = gr.Button("Segment (API)", variant="secondary")

        with gr.Row():
            api_mask = gr.File(label="mask.png")
            api_conf = gr.File(label="conf.png")
        api_json = gr.JSON(label="class_dist")

        api_btn.click(
            fn=api_segment,
            inputs=api_input,
            outputs=[api_mask, api_conf, api_json],
            api_name="segment",
        )

if __name__ == "__main__":
    demo.launch()
