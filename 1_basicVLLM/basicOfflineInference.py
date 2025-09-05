from vllm import LLM, SamplingParams

def main():
    # Define a list of input prompts
    prompts = [
        "The capital of France is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Initialize the LLM engine for CPU inference (see vLLM docs)
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",  # bfloat16 is recommended for CPU
        tensor_parallel_size=1,  # must be 1 for CPU
    )

    # Generate outputs for the input prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()