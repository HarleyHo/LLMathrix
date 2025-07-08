# config.py
from dataclasses import dataclass
from typing import Dict
import os


@dataclass
class Config:
    """Configuration for models, prompts, datasets, and API clients."""
    prompt_types: list = None
    models: Dict[str, str] = None
    datasets: Dict[str, tuple] = None
    judge_api_key: str = os.environ["JUDGE_API_KEY"]
    judge_base_url: str = os.environ["JUDGE_API_BASE_URL"]
    test_api_key: str = os.environ["BAILIAN_API_KEY"]
    test_base_url: str = os.environ["BAILIAN_API_BASE_URL"]
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
