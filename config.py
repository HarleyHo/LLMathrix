# config.py
from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    """Configuration for models, prompts, datasets, and API clients."""
    prompt_types: [] = None
    models: Dict[str, str] = None
    datasets: Dict[str, tuple] = None
    judge_api_key: str = "sk-Ncp7i7NTFB9kD90ah9a7iOxCowh9xAe94yTSTcXFZ2c9WYu6"
    judge_base_url: str = "https://api.nuwaapi.com/v1"
    test_api_key: str = "sk-862221a70b5a4731afcf137853d0208b"
    test_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dataset_dir: str = "datasets"
    checkpoint_dir: str = "checkpoints"
    eval_result_dir: str = "evaluation_results"

    def __post_init__(self):
        self.prompt_types = self.prompt_types or ["Direct", "COT", "Self-Refine", "Self-Consistency"]
        self.models = self.models or {
            "Qwen2.5-Math-1.5B": "qwen2.5-math-1.5b-instruct",
            "DeepSeek-R1-Qwen-1.5B": "deepseek-r1-distill-qwen-1.5b"
        }
        self.datasets = self.datasets or {
            "GSM8K": ("openai/gsm8k", "main", "test"),
            "MATH-500": ("HuggingFaceH4/MATH-500", None, "test"),
            "AIME_2024": ("HuggingFaceH4/aime_2024", None, "train")
        }
