# import sys
# sys.path.append('/mnt/bn/siqi-sparse-rl/mnt/siqi_nas/suffix_sspec/vllm-selfspec/vllm/suffix_cache/build')
# import _C
# from _C import SuffixTree, Candidate
# print(SuffixTree)

# tree = SuffixTree(64)

# tree.append(0,0)

# print(tree)


import sys
# sys.path.append('/mnt/bn/siqi-sparse-rl/mnt/siqi_nas/suffix_sspec/vllm-selfspec/suffix_cache/build')
# import _C
# from siqi_C import SuffixTree, Candidate
from arctic_inference.common.suffix_cache._C import SuffixTree, Candidate

# # 移除指定路径（替换为你要移除的路径）
# path_to_remove = "/mnt/bn/siqi-sparse-rl/mnt/siqi_nas/suffix_sspec/vllm-selfspec/suffix_cache/build"
# if path_to_remove in sys.path:
#     print(sys.path)
#     sys.path.remove(path_to_remove)

tree = SuffixTree(64)

tree.append(0,0)

print(tree)
