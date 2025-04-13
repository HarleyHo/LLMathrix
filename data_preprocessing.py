# data_preprocessing.py
import os
from config import Config
import pandas as pd
from datasets import load_dataset
import logging

logger = logging.getLogger("my_logger")


def preprocess_dataset(name: str, dataset_config: tuple) -> pd.DataFrame:
    """Preprocess a dataset into a standardized DataFrame."""
    logger.info(f"Preprocessing dataset: {name}")
    dataset = load_dataset(dataset_config[0], dataset_config[1], split=dataset_config[2])

    if name == "GSM8K":
        return pd.DataFrame({
            "Q": dataset["question"],
            "S": dataset["answer"],
            "A": [x.split("####")[-1].strip() for x in dataset["answer"]]
        })

    return pd.DataFrame({
        "Q": dataset["problem"],
        "S": dataset["solution"],
        "A": dataset["answer"]
    })


def save_preprocessed_datasets(config: 'Config') -> None:
    """Save preprocessed datasets to CSV files."""
    os.makedirs(config.dataset_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.eval_result_dir, exist_ok=True)

    for dataset_name, dataset_config in config.datasets.items():
        dataset_dir = os.path.join(config.dataset_dir, dataset_name)
        checkpoint_dir = os.path.join(config.checkpoint_dir, dataset_name)

        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        dataset_file = os.path.join(dataset_dir, f"{dataset_name}.csv")
        if os.path.exists(dataset_file):
            logger.info(f"Preprocessed data of {dataset_name} already exists at {dataset_file}, skip preprocessing.")
            continue

        df = preprocess_dataset(dataset_name, dataset_config)
        df.to_csv(dataset_file, index=False)
        logger.info(f"Saved preprocessed data of {dataset_name} to {dataset_file}.")
