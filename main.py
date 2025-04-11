# main.py
from openai import OpenAI
from config import Config
from data_processing import save_preprocessed_datasets
from evaluation import evaluate_dataset, save_evaluation_results
from utils import setup_logging


def main():
    """Main function to run the evaluation pipeline."""
    logger = setup_logging()

    # Initialize configuration
    config = Config()

    # Initialize API clients
    judge_client = OpenAI(api_key=config.judge_api_key, base_url=config.judge_base_url)
    test_client = OpenAI(api_key=config.test_api_key, base_url=config.test_base_url)

    # Preprocess datasets
    save_preprocessed_datasets(config)

    # Initialize results
    results = {model_name: {pt: {} for pt in config.prompt_types} for model_name in config.models}

    # Evaluate models
    for model_name, test_model in config.models.items():
        for prompt_type in config.prompt_types:
            for dataset_name in config.datasets:
                logger.info(f"Evaluating {model_name} with {prompt_type} on {dataset_name}...")
                accuracy, avg_length = evaluate_dataset(
                    dataset_name, prompt_type, test_client, test_model, judge_client, config, model_name
                )
                results[model_name][prompt_type][dataset_name] = {
                    "accuracy": accuracy,
                    "avg_response_length": avg_length
                }
                logger.info(f"Accuracy: {accuracy:.4f}, Avg Response Length: {avg_length:.2f} chars")
                save_evaluation_results(results, config)

    # Print summary
    logger.info("\nEvaluation Summary:")
    for model_name, model_data in results.items():
        logger.info(f"\nModel: {model_name}")
        for prompt_type, prompt_data in model_data.items():
            logger.info(f"  Prompt: {prompt_type}")
            for dataset_name, metrics in prompt_data.items():
                logger.info(
                    f"    {dataset_name} - Accuracy: {metrics['accuracy']:.4f}, "
                    f"Avg Length: {metrics['avg_response_length']:.2f} chars"
                )


if __name__ == "__main__":
    main()
