## Understanding vLLM

What is vLLM?
An inference server is software that helps an AI model make new conclusions based on its prior training. Inference servers feed the input requests through a machine learning model and return an output.

Why do people use vLLM?
vLLM allows organizations to “do more with less” in a market where the hardware needed for LLM-based applications comes with a hefty price tag.

Building cost-efficient and reliable LLM services requires significant computing power, energy resources, and specialized operational skills. These challenges effectively put the benefits of customized, deployment-ready, and more security-conscious AI out of reach for many organizations.

vLLM and PagedAttention, the algorithm it’s built on, aim to address these challenges by making more efficient use of the hardware needed to support AI workloads.
Model I am serving here is `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`

vLLM provides an OpenAI-compatible HTTP API server, allowing you to interact via standard OpenAI Completions and Chat endpoints. You can serve any supported model.

vllM could be served in Helm Charts as shown in this: https://docs.vllm.ai/en/v0.7.3/getting_started/examples/chart-helm.html

Other ways to serve LLMs in https://docs.vllm.ai/en/v0.7.3/getting_started/examples/examples_online_serving_index.html

Other ways to serve: 
API Spec: https://github.com/meta-llama/llama-api-python


VLLM_CPU_KVCACHE_SPACE=20 VLLM_CPU_NUM_OF_RESERVED_CPU=1 vllm serve google/gemma-3n-E4B-it --dtype=bfloat16