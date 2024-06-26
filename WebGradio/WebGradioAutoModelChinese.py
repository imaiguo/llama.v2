#
## 支持历史记录的 llama2 Lora中文微调Chinese合并模型能力演示
#
#
import os
import platform
import gradio as gr

from loguru import logger

from llama2_wrapper import LLAMA2_WRAPPER, get_prompt, get_prompt_for_dialog

if platform.system() == 'Windows':
    os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + r'D:\devtools\PythonVenv\llama2\Lib\site-packages\torch\lib'

MODEL_PATH = "/opt/Data/ModelWeight/FlagAlpha/Llama2-Chinese-7b-Chat"
# MODEL_PATH = "/opt/Data/ModelWeight/FlagAlpha/Llama2-Chinese-13b-Chat"

# 模型加载 hf格式为fb16精度
model = LLAMA2_WRAPPER(
	model_path = MODEL_PATH,
    load_in_8bit = False,
    backend_type = "transformers"
)

def deleteByStartAndEnd(s, star):
    if star in s:
        x1 = s.index(star)
        nextposition = x1 + 1
        # print(nextposition)
        # print(s[nextposition:len(s)])
        if star in s[nextposition:len(s)]:
            x2 = s.index(star, nextposition) + len(star)
            x3 = s[x1:x2]
            # print(x3)
            result = s.replace(x3, "")
            result = deleteByStartAndEnd(result, star)
            return result
    return s

def predict(chatbot, history):
    input = chatbot[-1][0]
    logger.debug(f"input->:{input}")
    history += [{"role":"user", "content":input}]
    prompt = get_prompt_for_dialog(history)

    for response in model.generate(prompt, max_new_tokens=4096, temperature=1.0):
        chatbot[-1][1] = deleteByStartAndEnd(response, "*")
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

    begin = [{"role":"system", "content":"你是智能客服小蓝，仔细分析用户的输入并作详细又准确的回答。"}]
    history = gr.State(value=begin)

    subMsg = submitBtn.click(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)
    inputMsg = user_input.submit(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)

    subMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)
    inputMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)

auth=[("llm","llm123456"),("zhangsan","zhangsan123")]

demo.queue().launch(server_name="192.168.2.200", server_port=8000, inbrowser=False, share=False, auth=auth)