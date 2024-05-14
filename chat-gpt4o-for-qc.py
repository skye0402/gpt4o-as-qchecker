import gradio as gr
from openai import OpenAI

from typing import Optional
import base64

def perform_qc_check(query: str, ref_image: str, chk_image: str, detail_level: Optional[str]="low")->str:
    """ Performs the image conversion and comparison """
    client = OpenAI()
    base64_reference_image = encode_image(image_path=ref_image)
    base64_check_image = encode_image(image_path=chk_image)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a quality check expert in a factory. Your task is to find deviations between a reference product assortments and the one to be shipped to a customer.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_reference_image}",
                            "detail": detail_level,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_check_image}",
                            "detail": detail_level,
                        },
                    },
                ],
            },
        ],
        max_tokens=300,
    )
    return response.model_dump_json()

def encode_image(image_path):    
    """ Encodes the image to base 64 """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_quality(reference_image, check_image, text_input):
    if reference_image and check_image and text_input:
        response = perform_qc_check(query=text_input, ref_image=reference_image, chk_image=check_image)
        return f"Quality check complete for: {text_input}"
    else:
        return ""

def create_page(default_prompt: str)->gr.Blocks:
    with gr.Blocks() as qc_app:
        gr.Markdown("# Quality Check scenario using OpenAI GPT-4o")
        output_textbox = gr.Textbox(label="Check Result by AI", interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Reference Image")
                reference_image = gr.Image(label="Upload Reference Image", type="filepath")
            with gr.Column():
                gr.Markdown("## Q-Check Image")
                quality_image = gr.Image(label="Upload Quality Check Image", type="filepath")
        
        gr.Markdown("## Input Field")
        with gr.Row():
            text_input = gr.Textbox(label="Enter text here", value=default_prompt)
            check_button = gr.Button("Check")
        
        check_button.click(
            fn=check_quality,
            inputs=[reference_image, quality_image, text_input],
            outputs=output_textbox
        )
    return qc_app

def main()->None:
    """ Main function of the chat server """
    q_tip_prompt = """The first image is the reference. Compare with second image (the image to be checked against) and respond with:\n 
    1. How many Q-Tips are in the reference versus the check image?
    2. Are the Q-Tips arranged similar to the reference? Pay attention to the orientation. If so, state the number(s) of the according Q-Tip.
    3. Are there any damages to the Q-Tip or impurities? If so, state the number(s) of the according Q-Tip.
    4. Your general comment if you would ship the check image products to the customer."""
    qc_check_page = create_page(default_prompt=q_tip_prompt)
    qc_check_page.launch(
        debug=False,
        show_api=False,
        allowed_paths=["./img"]
    )
    
if __name__ == "__main__":
    main()