import gradio as gr

from predict import Predictor

detector = Predictor()


def set_model(model_file):
    try:
        detector.load_model(model_file.name)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict(image):
    try:
        result = detector.predict(image, batch_size=1)
        return result[0]

    except Exception as e:
        return {"result": "failed", 'error': str(e)}


demo = gr.Blocks()

with demo:
    gr.Markdown("Inference demo")
    with gr.Tabs():
        with gr.TabItem("Set model"):
            text_input = gr.File()
            text_output = gr.Text()
            text_button = gr.Button("Set")
        with gr.TabItem("Inference"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Text()
            image_button = gr.Button("Predict")

    text_button.click(set_model, inputs=text_input, outputs=text_output)
    image_button.click(predict, inputs=image_input, outputs=image_output)

demo.launch(share=True)
