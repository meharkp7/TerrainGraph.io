# app.py — HuggingFace Space demo entrypoint (Gradio).
# Must stay in sync with the GitHub repo: Spaces deploy from this file.
import gradio as gr
from pipeline_hf import run_pipeline
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


with gr.Blocks(theme=gr.themes.Soft(), title="TerrainGraph AI") as demo:
    gr.Markdown("""
    # TerrainGraph AI
    ### Intelligent Terrain Analysis & Safe Navigation
    DeepLabV3+ segmentation → Dijkstra safe-path planning → AI briefing.
    Full system (TigerGraph memory + LLM reasoning + React dashboard) on
    [GitHub](https://github.com/meharkp7/TerrainGraph.io).
    """)

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

if __name__ == "__main__":
    demo.launch()
