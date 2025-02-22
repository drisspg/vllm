# SPDX-License-Identifier: Apache-2.0
# quantization_method.py
from vllm import LLM, SamplingParams


def main():
    # Create the quantization config
    # quant_config = TorchAOConfig(
    #     bit_width=8,
    #     use_symmetric=True
    # )

    # Initialize the LLM with quantization
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",  # Or your model path
        quantization="torchao",
        trust_remote_code=True,
        dtype="float16",  # Use FP16 for efficiency
        tensor_parallel_size=1,
        max_num_seqs=1,  # Process 1 sequence at a time
        max_num_batched_tokens=4096
    )

    # Create sampling parameters
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=100)

    # Test the model
    prompts = [
        "Write a short poem about coding:",
        "Explain quantum computing in one sentence:"
    ]

    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")


if __name__ == "__main__":
    main()
