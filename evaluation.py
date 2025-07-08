# evaluation.py
import os
import re
import pandas as pd
from typing import Tuple, Dict, Optional
from openai import OpenAI
import logging
from config import Config

logger = logging.getLogger("my_logger")


def judge_response(
        response: Optional[str],
        true_answer: str,
        judge_client: OpenAI,
        judge_model: str = "gpt-4o",
        temperature: float = 0
) -> tuple[bool, bool]:
    """
    Judge if a response is equivalent to the true answer, first using pattern matching, then GPT-4o if needed.

    Args:
        temperature:
        response (Optional[str]): Generated response.
        true_answer (str): Ground truth answer.
        judge_client (OpenAI): GPT-4 API client.
        judge_model (str): GPT-4 model name.

    Returns:
        tuple[bool, bool]: First bool for True if response is equivalent to true answer, False otherwise.
                                  Second bool for True if judge LLM is used, False otherwise.
    """
    if not response:
        logger.debug("Judgment: Response is None, returning False.")
        return False, False

    # Step 1: Pattern matching
    true_answer_str = str(true_answer).strip()

    # Extract numerical or symbolic answer from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    predicted_answer = numbers[-1].strip() if numbers else response.strip()

    # Normalize answers for comparison
    normalized_predicted = predicted_answer.replace(" ", "").lower()
    normalized_true = true_answer_str.replace(" ", "").lower()

    if normalized_predicted == normalized_true:
        logger.debug(f"Pattern matching succeeded: Predicted '{predicted_answer}' matches true '{true_answer_str}'.")
        return True, False

    logger.info(
        f"Pattern matching failed: Predicted '{predicted_answer}' does not match true '{true_answer_str}'. "
        f"Falling back to Judge LLM {judge_model}."
    )

    # Step 2: Judge LLM judgment
    system_prompt = (
        "You are a precise mathematical judge. Your task is to determine if two answers are mathematically equivalent."
    )
    user_prompt = (
        f"Generated Answer:\n{response}\n"
        f"True Answer:\n{true_answer}\n"
        f"Are these answers mathematically equivalent?"
        f"Answer briefly: 'True' for equivalent, 'False' for not equivalent."
    )

    from inference import generate_completion
    judge_result = generate_completion(judge_client, judge_model, system_prompt, user_prompt, temperature)
    if judge_result is None:
        logger.error("GPT-4o judgment failed: No result returned.")
        return False, True

    result = judge_result.strip().lower() == "true"
    logger.debug(f"GPT-4o judgment result: {result} (Response: {response}, True: {true_answer})")
    return result, True


def evaluate_dataset(
        dataset_name: str,
        prompt_type: str,
        test_client: OpenAI,
        test_model: str,
        judge_client: OpenAI,
        config: 'Config',
        model_name: str
) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset with a specific prompt type.

    Args:
        dataset_name (str): Name of the dataset.
        prompt_type (str): Prompt type (COT, Self-Refine, Self-Consistency).
        test_client (OpenAI): API client for testing.
        test_model (str): Model name for testing.
        judge_client (OpenAI): API client for judging.
        config (Config): Configuration object.
        model_name (str): Name of the model.

    Returns:
        Tuple[float, float]: Accuracy and average response length.
    """
    dataset_dir = os.path.join(config.dataset_dir, dataset_name)
    input_file = os.path.join(dataset_dir, f"{dataset_name}.csv")
    output_file = os.path.join(dataset_dir, f"{dataset_name}_{model_name}_{prompt_type}.csv")

    checkpoint_dir = os.path.join(config.checkpoint_dir, dataset_name)
    checkpoint_file = os.path.join(checkpoint_dir, f"{dataset_name}_{model_name}_{prompt_type}_checkpoint.txt")

    try:
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
        else:
            df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Dataset file {input_file} not found.")
        return 0.0, 0.0

    response_col = f"Response_{model_name}_{prompt_type}"
    correct_col = f"Is_Correct_{model_name}_{prompt_type}"
    length_col = f"Response_Length_{model_name}_{prompt_type}"

    for col in [response_col, correct_col, length_col]:
        if col not in df.columns:
            df[col] = "" if col.startswith("Response") else False if col.startswith("Is_Correct") else 0

    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_idx = int(f.read().strip())

    from inference import generate_response
    for idx in range(start_idx, len(df)):
        question = df.at[idx, "Q"]
        true_answer = str(df.at[idx, "A"]).strip()

        response = generate_response(
            question, prompt_type, test_client, test_model, judge_client=judge_client
        )
        is_correct, used_llm_judge = judge_response(response, true_answer, judge_client)

        logger.debug(f"Response:\n{response}\nAnswer:{true_answer}\nCorrect:{is_correct}")

        df.at[idx, response_col] = response or ""
        df.at[idx, correct_col] = is_correct
        df.at[idx, length_col] = len(response) if response else 0

        df.to_csv(output_file, index=False)
        with open(checkpoint_file, "w") as f:
            f.write(str(idx + 1))

        logger.info(
            f"Processed {idx + 1}/{len(df)}, Dataset: {dataset_name}, "
            f"Model: {model_name}, Prompt: {prompt_type}, Result: {is_correct}, Used LLM Judge: {used_llm_judge}"
        )

    accuracy = df[correct_col].mean()
    avg_length = df[length_col].mean()
    return accuracy, avg_length


def save_evaluation_results(results: Dict, config: 'Config', file_name="results") -> None:
    """Save evaluation results to a CSV file."""
    result_rows = []

    for model_name, model_data in results.items():
        for prompt_type, prompt_data in model_data.items():
            for dataset_name, metrics in prompt_data.items():
                result_rows.append({
                    "Model": model_name,
                    "Prompt": prompt_type,
                    "Dataset": dataset_name,
                    "Accuracy": metrics["accuracy"],
                    "Avg_Response_Length": metrics["avg_response_length"]
                })

    df_results = pd.DataFrame(result_rows)
    output_file = os.path.join(config.eval_result_dir, f"{file_name}.csv")

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = (pd.concat([existing_df, df_results]).drop_duplicates(subset=["Model", "Prompt", "Dataset"]))
        combined_df.to_csv(output_file, index=False)
    else:
        df_results.to_csv(output_file, index=False)

    logger.info(f"Saved evaluation results successfully.")
