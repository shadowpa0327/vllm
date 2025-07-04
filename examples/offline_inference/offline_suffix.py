# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import vllm
from vllm import LLM, SamplingParams

import os

if __name__ == '__main__':
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    vllm.plugins.load_general_plugins()

    llm = LLM(
        model="/mnt/bn/siqi-sparse-rl/mnt/siqi_nas/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen2.5-3B-Instruct",
        # quantization="fp8",
        tensor_parallel_size=1,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 3,
            "model": None,
            # "enable_suffix_decoding": True,
            "disable_by_batch_size": 64,
        },
        seed=0,
    )

    print("=" * 80)

    conversation = [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]

    sampling_params = SamplingParams(temperature=0.1, max_tokens=128)

    outputs = llm.chat(conversation, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)
