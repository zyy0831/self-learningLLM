from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class myLLM(LLM):
    # 基于本地 Qwen 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: Qwen 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        # model_dir = 'model\Shanghai_AI_Laboratory\internlm2-chat-1_8b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        response, history = self.model.chat(self.tokenizer, prompt , history=[])
        return response
        
    @property
    def _llm_type(self) -> str:
        return "myLLM"