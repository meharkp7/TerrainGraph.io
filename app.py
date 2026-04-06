#   FOR HUGGING FACE SPACE DEMO
import gradio as gr
from pipeline_hf import run_pipeline
from utils import create_bar_chart

def predict(image):
    result = run_pipeline(image)
    chart = create_bar_chart(result["class_dist"])

    return (
        result["original"],
        result["overlay"],
        result["briefing"],
        chart
    )

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌍 TerrainGraph AI  
    ### Intelligent Terrain Analysis & Safe Navigation
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    run_btn = gr.Button("🚀 Analyze Terrain")

    status = gr.Markdown("")

    with gr.Row():
        original = gr.Image(label="Original")
        overlay = gr.Image(label="Safe Path")

    briefing = gr.Textbox(label="🧠 AI Briefing")

    chart = gr.Plot(label="📊 Terrain Distribution")

    def wrapped_predict(img):
        return predict(img)

    run_btn.click(
        fn=wrapped_predict,
        inputs=image_input,
        outputs=[original, overlay, briefing, chart]
    )

demo.launch()