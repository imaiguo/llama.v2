
import os
import platform
import gradio as gr

from loguru import logger

from llama2_wrapper import LLAMA2_WRAPPER, get_prompt, get_prompt_for_dialog

if platform.system() == 'Windows':
    os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + r'D:\devtools\PythonVenv\llama2\Lib\site-packages\torch\lib'

if platform.system() == "Windows":
    MODEL_PATH = "E:/THUDM/llama2/model/llama-2-7b-chat"
else:
    MODEL_PATH = "/opt/Data/THUDM/llama2.hf/llama-2-7b-chat-hf"

model = LLAMA2_WRAPPER(
	model_path = MODEL_PATH,
    backend_type = "transformers"
)

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def predict(chatbot, history):
    input = chatbot[-1][0]
    logger.debug(f"input->:{input}")
    history += [{"role":"user", "content":input}]
    prompt = get_prompt_for_dialog(history)

    for response in model.generate(prompt, max_new_tokens=4096, temperature=1.0):
        chatbot[-1][1] = parse_text(response)
        yield chatbot, history

    history += [{"role":"assistant", "content":response }]

    logger.debug(f"history->:{history}")

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

with gr.Blocks(title = "Llama2", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=800,
        show_copy_button = False,
        layout= "bubble",
        avatar_images=("./image/Einstein.jpg", "./image/openai.png")
    )

    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

        with gr.Column(min_width=1, scale=1):
            submitBtn = gr.Button("提交", variant="primary")

    begin = [{"role":"system", "content":"You are Intelligent Customer Service Blue, carefully analyzing the user's input and providing detailed and accurate answers."}]
    history = gr.State(value=begin)

    subMsg = submitBtn.click(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)
    inputMsg = user_input.submit(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)

    subMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)
    inputMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)

auth=[("llm","llm123456"),("zhangsan","zhangsan123")]

if platform.system() == 'Windows':
    demo.queue().launch(server_name="192.168.2.198", server_port=8001, inbrowser=False, share=False, auth=auth)
else:
    demo.queue().launch(server_name="192.168.2.200", server_port=8000, inbrowser=False, share=False, auth=auth)