import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
# model_dir = snapshot_download('qwen/Qwen1.5-0.5B', cache_dir='E:\LLM\model', revision='master')
model_dir = snapshot_download('Jerry0/m3e-base', cache_dir='E:\LLM\model', revision='master')