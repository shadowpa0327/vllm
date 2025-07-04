# csrc/custom_ops可以忽略，不会用到

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate

export VLLM_COMMIT=5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-0.9.0.1-cp38-abi3-manylinux1_x86_64.whl
uv pip install -e .

uv pip install --no-build-isolation "git+https://github.com/flashinfer-ai/flashinfer@21ea1d2545f74782b91eb8c08fd503ac4c0743fc"

# 安装suffix tree 到vllm-selfspec/build/lib.linux-x86_64-cpython-311/arctic_inference/common/suffix_cache/_C.cpython-311-x86_64-linux-gnu.so
python setup1.py install
# from arctic_inference.common.suffix_cache._C import SuffixTree, Candidate
# see vllm-selfspec/test.py

export TORCH_CUDA_ARCH_LIST="8.9+PTX"
python /mnt/bn/siqi-sparse-rl/mnt/siqi_nas/suffix_sspec/vllm-selfspec/examples/offline_inference/offline_suffix.py