

```sh
python3 -m venv aienv
source aienv/bin/activate

# VLLM REQUIREMENTS: https://docs.vllm.ai/en/v0.7.3/getting_started/installation/cpu/index.html
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

# install python packages and install vLLM

login with Hugging Face to download your desired model weights

# https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.2/html/vllm_server_arguments/vllm-server-usage_server-arguments
```

Show Everyone the architecture 

```sh
lscpu
```