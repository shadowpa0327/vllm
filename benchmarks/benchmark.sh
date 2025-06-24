

python3 benchmarks/benchmark_throughput.py   --model /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32 --enable-speculative
python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen2.5-7B-Instruct   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32


python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/heritage/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32 --enable-speculative
python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/heritage/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32


python3 benchmarks/benchmark_throughput.py   --model /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-4B   --backend vllm   --dataset-name hf   --dataset-path AI-MO/aimo-validation-aime   --hf-split train   --num-prompts 32 --enable-speculative
