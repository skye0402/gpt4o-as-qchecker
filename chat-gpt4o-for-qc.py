from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import gradio as gr
from openai import OpenAI

from typing import Optional
import base64, json

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
        
        response = '{"id":"chatcmpl-9OaUYJHZSbzN3jSFi0fppVA53k5az","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"1. The reference image contains 11 Q-Tips, while the check image also contains 11 Q-Tips.\\n\\n2. Yes, some Q-Tips are arranged similarly to the reference. The Q-Tips in positions G, H, I, K, L, and M are arranged in a similar orientation as the reference image.\\n\\n3. Yes, there are impurities and damages noticed:\\n   - The Q-Tips in positions A and C are bent and not straight.\\n   - The Q-Tip in position E has a black impurity on it.\\n   - The Q-Tip in position J has a green impurity on it.\\n\\n4. Based on the impurities and damages noted, I would not ship the check image products to the customer. The presence of bent Q-Tips and impurities would not meet the quality standards expected for shipment.","role":"assistant","function_call":null,"tool_calls":null}}],"created":1715647174,"model":"gpt-4o-2024-05-13","object":"chat.completion","system_fingerprint":"fp_927397958d","usage":{"completion_tokens":171,"prompt_tokens":341,"total_tokens":512}}'
        response = '{"id":"chatcmpl-9ObN26dFCzIodoc99n0eS4RB3GLeq","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"1. **Number of Q-Tips:**\\n   - Reference Image: 11 Q-Tips\\n   - Check Image: 12 Q-Tips\\n\\n2. **Q-Tip Arrangement Similarity:**\\n   - Q-Tips in the following positions are arranged similarly to the reference: F, G, I, J, and K.\\n\\n3. **Q-Tip Damages or Impurities:**\\n   - Q-Tip in position E is broken.\\n   - Q-Tip in position H has a green impurity.\\n\\n4. **General Comment:**\\n   - The check image shows discrepancies in the number of Q-Tips compared to the reference.\\n   - There are serious issues such as a broken Q-Tip in position E and an impurity on the Q-Tip in position H.\\n   - Considering these issues, I would recommend not shipping the products to the customer until the defective Q-Tips are replaced and the correct number of Q-Tips is ensured.","role":"assistant","function_call":null,"tool_calls":null}}],"created":1715650552,"model":"gpt-4o-2024-05-13","object":"chat.completion","system_fingerprint":"fp_927397958d","usage":{"completion_tokens":199,"prompt_tokens":379,"total_tokens":578}}'
        response = perform_qc_check(query=text_input, ref_image=reference_image, chk_image=check_image)
        print(response)
        res_json = json.loads(response)
        
        return f"""## QC Result from AI:\n{res_json["choices"][0]["message"]["content"]}"""
    else:
        return """"## QC Result from AI:"""

def create_page(default_prompt: str)->gr.Blocks:
    with gr.Blocks() as qc_app:
        gr.Markdown("# 🕵️ Quality Check scenario using OpenAI GPT-4o 🔬")
        qc_result = gr.Markdown("""## QC Result from AI:""")
        output_textbox = gr.Textbox(show_label=False, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🆗 Reference Image")
                reference_image = gr.Image(label="Upload Reference Image", type="filepath")
            with gr.Column():
                gr.Markdown("## 🔍 Q-Check Image")
                quality_image = gr.Image(label="Upload Quality Check Image", type="filepath", mirror_webcam=False)
        
        gr.Markdown("## ✍️ Quality Check Instructions to GPT-4o")
        text_input = gr.Textbox(label="Enter text here", value=default_prompt)
        check_button = gr.Button("👀 Check")
        
        check_button.click(
            fn=check_quality,
            inputs=[reference_image, quality_image, text_input],
            outputs=qc_result
        )
    return qc_app

def main()->None:
    """ Main function of the chat server """
    q_tip_prompt = """The first image is the reference. Compare with second image (the image to be checked against) and respond with:\n 
    1. How many Q-Tips are in the reference versus the check image?
    2. Are the Q-Tips arranged similar to the reference? Pay attention to the orientation. If so, state the number(s) of the according Q-Tip.
    3. Are there any damages to the Q-Tip or impurities? If so, state the number(s) of the according Q-Tip.
    4. Your general comment if you would ship the check image products to the customer.
    Note: As long as a Q-Tip is inside the black box it's ok if the orientation is a bit to the left or right or rotated. Such cases are counted as good."""
    qc_check_page = create_page(default_prompt=q_tip_prompt)
    qc_check_page.launch(
        debug=False,
        show_api=False,
        allowed_paths=["./img"]
    )
    
if __name__ == "__main__":
    main()