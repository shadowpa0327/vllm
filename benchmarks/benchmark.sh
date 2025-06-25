

python3 benchmarks/benchmark_throughput.py   --model /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 64 --enable-speculative
python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen2.5-7B-Instruct   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32


python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/heritage/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32 --enable-speculative
python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/heritage/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32


python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-4B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32 --enable-speculative
