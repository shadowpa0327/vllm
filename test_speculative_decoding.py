from vllm import LLM, SamplingParams
import os

#os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

prompts = ["The future of AI is", "In the world of technology,"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20, ignore_eos=True)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")