
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn, json, datetime
import torch
import platform

if platform.system() == "Windows":
    MODEL_PATH = "E:/THUDM/llama2.hf/llama-2-7b-chat-hf"
else:
    MODEL_PATH = "/opt/Data/THUDM/llama2.hf/llama-2-7b-chat-hf"

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI()

B_INST, E_INST = "[INST]", "[/INST]"

SYSTEM = """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"""

@app.post("/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    print(json_post_list)
    prompt = json_post_list.get('messages')
    # print(prompt)
    # print(prompt[0].get('role'))
    content = prompt[0].get('content')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    if history is None or len(history) == 0:
        prompt = SYSTEM + content.strip()
        history = []

    input_ids = []
    for q, a in history:
        input_ids += tokenizer.encode(f"{B_INST} {q} {E_INST} {a}") + [tokenizer.eos_token_id]

    input_ids += tokenizer.encode(f"{B_INST} {prompt} {E_INST}")

    response = model.generate(torch.tensor([input_ids]).cuda(),
                                   max_length=max_length if max_length else 4096,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)

    response = tokenizer.decode(response[0, len(input_ids):], skip_special_tokens=True)
    history.append([prompt, response])

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }

    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()

    return answer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map='cuda'
        )

    model.eval()
    if platform.system() == 'Windows':
        uvicorn.run(app, host='192.168.2.198', port=9001, workers=1)
    else:
        uvicorn.run(app, host='192.168.2.200', port=9000, workers=1)