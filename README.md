
# LLAMA2


Llama1 Content Length: 2K
Llama2 Content Length: 4K
Llama3 Content Length: 8K

## Windows环境部署

 准备独立的python环境

```bash
> cmd
> cd D:\devtools\PythonVenv
> python3 -m venv llama2
> D:\devtools\PythonVenv\llama2\Scripts\activate.bat
> set http_proxy=192.168.2.199:58591
> set http_proxy=192.168.2.199:58591
```

部署推理环境

```bash
> pip install -r requirements.txt
> pip install jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

python环境切换
```bash
> cmd
> D:\devtools\PythonVenv\llama2\Scripts\activate.bat
```

vscode切换
使用CTRL+shift+p命令: 打开命令交互面板， 在命令面板中可以输入命令进行搜索, 输入 D:\devtools\PythonVenv\llama2
```bash
> Python: Select Interpreter
```

```bash
> torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
>
> torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir E:\THUDM\llama2\model\llama-2-7b-chat --tokenizer_path E:\THUDM\llama2\model\tokenizer.model --max_seq_len 128 --max_batch_size 4
```

```bash
>
> python example_text_completion02.py --ckpt_dir E:\THUDM\llama2\model\llama-2-7b-chat --tokenizer_path E:\THUDM\llama2\model\tokenizer.model --max_seq_len 128 --max_batch_size 4
>
```

## Debian环境部署

设置代理环境
```bash
> export http_proxy=192.168.2.199:58591
> export https_proxy=192.168.2.199:58591
```

安装cuda

```bash
> wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb
> sudo dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb
> sudo cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
> sudo add-apt-repository contrib
> sudo apt-get update
> sudo apt-get -y install cuda-toolkit-12-3
```

设置python虚拟环境
```bash
> sudo apt install python3-venv  python3-pip
> mkdir /opt/Data/PythonVenv
> cd /opt/Data/PythonVenv
> python3 -m venv llama2
> source /opt/Data/PythonVenv/llama2/bin/activate
```

部署推理环境

```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install gradio
> pip install pyreadline3
> pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install transformers==4.31.0 protobuf accelerate llama2-wrapper -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install llama2-wrapper -i https://pypi.tuna.tsinghua.edu.cn/simple
> 
```

## 模型参数格式转换

bfb16转fb16 [pytorch训练格式 -> hugging face格式]

```bash
> # Linux
> python llama-recipes/src/llama_recipes/tools/convert_llama_weights_to_hf.py \
    --input_dir /opt/Data/THUDM/llama2 --model_size 7B --output_dir /opt/Data/THUDM/llama2.hf/llama-2-7b-chat-hf
>
> python llama-recipes/src/llama_recipes/tools/convert_llama_weights_to_hf.py --input_dir /opt/Data/THUDM/llama2 --model_size 13B --output_dir /opt/Data/THUDM/llama2.hf/llama-2-13b-chat-hf
>
> # windows
> python llama-recipes/src/llama_recipes/tools/convert_llama_weights_to_hf.py \
    --input_dir E:/THUDM/llama2 --model_size 7B --output_dir E:/THUDM/llama2.hf/llama-2-7b-chat-hf
>
```

启动服务
```bash
> python openai/openapi.py
> python WebGradio/WebGradioAutoModel.py
> python WebGradio/WebGradioAutoModelLora.py
> python WebGradio/WebGradioAutoModelAtom.py
> jupyter notebook --no-browser --port 7001 --ip=192.168.2.198
> jupyter notebook --no-browser --port 7000 --ip=192.168.2.200
```