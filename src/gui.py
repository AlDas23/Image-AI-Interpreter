import os
import gradio as gr
import src.backend as backend


def get_model_list():
    model_path = "models/"
    if not os.path.exists(model_path):
        return []  # Return empty list if folder doesn't exist
    return [
        name
        for name in os.listdir(model_path)
        if (os.path.isdir(os.path.join(model_path, name))
            or os.path.isfile(os.path.join(model_path, name)))
        and name != "put_models_here"
    ]


def refresh_models():
    return gr.Dropdown.update(choices=get_model_list())


def generate_response(modelpath, prompt, img):
    response = backend.generate_response(modelpath, prompt, img)
    return response


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy", sources=["upload"], label="Input Image"
            )
            prompt_input = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here"
            )
            submit_btn = gr.Button("Submit")
        with gr.Column():
            with gr.Row():
                model_selector = gr.Dropdown(
                    label="Model selection",
                    choices=get_model_list(),
                    value=None,
                )
                refresh_btn = gr.Button("Refresh")
            with gr.Row():
                output_text = gr.Textbox(
                    label="Output", placeholder="Output will be shown here"
                )

    refresh_btn.click(refresh_models, outputs=model_selector)

    submit_btn.click(
        generate_response,
        inputs=[model_selector, prompt_input, image_input],
        outputs=output_text,
    )
