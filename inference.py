# inference.py
import json
import re
from typing import Optional
from collections import Counter
from openai import OpenAI
import logging

logger = logging.getLogger("my_logger")


def generate_completion(
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
) -> Optional[str]:
    """Generate a completion using the specified client and model."""
    try:
        # noinspection PyTypeChecker
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=True
        )

        collected_messages = []
        # iterate through the stream of events
        for chunk in completion:
            logger.debug(f"Chunk: {chunk}")
            chunk_message = chunk.choices[0].delta.content  # extract the message
            if chunk_message is not None:
                collected_messages.append(chunk_message)  # save the message

        # print the time delay and text received
        full_reply_content = ''.join(collected_messages)
        return full_reply_content
    except Exception as e:
        logger.error(f"Failed to generate completion: {e}")
        # raise CompletionError(f"Failed to generate completion: {e}") from e
        return None


def direct_answer(question: str, client: OpenAI, model: str, temperature: float = 0) -> Optional[str]:
    """Generate a direct answer without any prompting strategy."""
    system_prompt = "You are a helpful assistant skilled in solving math problems."
    user_prompt = (
        f"Problem: {question}\n"
        f"Provide the solution of the problem."
    )
    response = generate_completion(client, model, system_prompt, user_prompt, temperature)
    if response:
        logger.debug(f"Direct response for question '{question[:50]}...': {response}")
    return response


def chain_of_thought(question: str, client: OpenAI, model: str, temperature: float = 0) -> Optional[str]:
    """
    Generate an answer using Chain-of-Thought (COT) prompting.

    Args:
        question (str): The math problem to solve.
        client (OpenAI): API client.
        model (str): Model name.

    Returns:
        Optional[str]: Generated answer or None if failed.
    """
    system_prompt = "You are a helpful assistant skilled in solving math problems step by step."
    user_prompt = (
        f"Problem: {question}\n"
        f"Solve the problem step by step."
    )
    response = generate_completion(client, model, system_prompt, user_prompt, temperature)

    if response:
        logger.debug(f"COT response for question '{question[:50]}...': {response}")
    return response


def self_refine(question: str, client: OpenAI, model: str, temperature: float = 0) -> Optional[str]:
    """
    Execute a three-stage Self-Refine process.

    Args:
        temperature:
        question (str): The math problem to solve.
        client (OpenAI): API client.
        model (str): Model name.

    Returns:
        Optional[str]: Final refined answer or None if any stage fails.
    """
    try:
        # Stage 1: Initial Solution
        initial_prompt = (
            f"Problem: {question}\n"
            f"Provide a detailed initial solution step by step. "
        )
        initial_response = generate_completion(
            client=client,
            model=model,
            system_prompt="You are a helpful assistant skilled in solving math problems.",
            user_prompt=initial_prompt,
            temperature=temperature
        )
        if not initial_response:
            logger.error("Initial stage failed: No response generated.")
            return None
        logger.debug(f"Self-Refine Initial response: {initial_response}")

        # Stage 2: Feedback
        feedback_prompt = (
            f"Initial Solution:\n{initial_response}\n"
            f"Problem: {question}\n"
            f"Please review the initial solution for errors, inconsistencies, or areas for improvement. "
            f"Provide specific feedback to improve the solution."
        )
        feedback_response = generate_completion(
            client=client,
            model=model,
            system_prompt="You are a critical reviewer of math solutions.",
            user_prompt=feedback_prompt,
            temperature=temperature
        )
        if not feedback_response:
            logger.error("Feedback stage failed: No feedback generated.")
            return None
        logger.debug(f"Self-Refine Feedback: {feedback_response}")

        # Stage 3: Refine
        refine_prompt = (
            f"Initial Solution:\n{initial_response}\n"
            f"Feedback:\n{feedback_response}\n"
            f"Problem: {question}\n"
            f"Based on the feedback, provide a refined solution. Ensure the answer is clear and correct."
        )
        refined_response = generate_completion(
            client=client,
            model=model,
            system_prompt="You are a helpful assistant refining a math solution.",
            user_prompt=refine_prompt,
            temperature=temperature
        )
        if not refined_response:
            logger.error("Refine stage failed: No refined response generated.")
            return None
        logger.debug(f"Self-Refine Final response: {refined_response}")

        return refined_response

    except Exception as e:
        logger.error(f"Error in Self-Refine: {e}")
        return None


def self_consistency(
        question: str,
        client: OpenAI,
        model: str,
        judge_client: OpenAI,
        judge_model: str = "gpt-4o",
        num_attempts: int = 5,
        temperature: float = 1.5,
        judge_temperature: float = 0
) -> Optional[str]:
    """
    Generate an answer using Self-Consistency: generate multiple COT responses and vote for the final answer.

    Args:
        temperature:
        judge_temperature:
        question (str): The math problem to solve.
        client (OpenAI): API client for generating responses.
        model (str): Model name for generating responses.
        judge_client (OpenAI): API client for voting.
        judge_model (str): Model name for voting.
        num_attempts (int): Number of COT attempts.

    Returns:
        Optional[str]: Final answer selected by voting or None if failed.
    """
    try:
        # Generate multiple COT responses
        responses = []
        for attempt in range(num_attempts):
            response = chain_of_thought(question, client, model, temperature)
            if response:
                responses.append(response)
                logger.debug(f"Self-Consistency attempt {attempt + 1}: {response}")
            else:
                logger.warning(f"Self-Consistency attempt {attempt + 1} failed.")

        if not responses:
            logger.error("Self-Consistency failed: No valid responses generated.")
            return None

        # Save responses and selected responses to file
        save_to_json(question, responses)

        # If only one response, save and return it
        if len(responses) == 1:
            logger.info("Self-Consistency: Only one response generated, returning it.")
            return responses[0]

        # Use GPT-4o to vote for the most consistent answer
        voting_prompt = (
                f"Multiple solutions were generated:\n"
                + "\n".join([f"Solution {i + 1}: {resp}" for i, resp in enumerate(responses)])
                + "\nPlease analyze these solutions to identify the answer that appears most frequently. "
                  "Return the index of the first solution that contains the most frequent answer in the format: **Final Answer**: Solution [Your chosen index]\n"
        )
        voting_response = generate_completion(
            client=judge_client,
            model=judge_model,
            system_prompt=(
                "You are a precise mathematical judge. Your task is to analyze multiple solutions to a problem "
                "and select the most consistent and correct answer."
            ),
            user_prompt=voting_prompt,
            temperature=judge_temperature
        )

        selected_response = None

        if not voting_response:
            logger.error("Self-Consistency voting failed: No response from voting LLM.")
            # Fallback: select the most common response
            most_common = Counter(responses).most_common(1)
            if most_common:
                selected_response = most_common[0][0]
        else:
            logger.debug(f"Self-Consistency voting result: {voting_response}")
            match = re.search(r"\*\*Final Answer\*\*: Solution (\d+)", voting_response)
            if match:
                number = int(match.group(1)) - 1
            else:
                number = 0
            selected_response = responses[number]

        # Fallback if no specific solution was selected
        if not selected_response:
            selected_response = responses[0]

        # Return the first selected response
        return selected_response

    except Exception as e:
        logger.error(f"Error in Self-Consistency: {e}")
        return None


def generate_response(
        question: str,
        prompt_type: str,
        client: OpenAI,
        model: str,
        judge_client: OpenAI = None,
        judge_model: str = "gpt-4o"
) -> Optional[str]:
    """
    Generate a response using the specified prompt type.

    Args:
        question (str): The input question.
        prompt_type (str): Type of prompt (COT, Self-Refine, Self-Consistency).
        client (OpenAI): API client for generating responses.
        model (str): Model name.
        judge_client (OpenAI, optional): API client for judging (required for Self-Consistency).
        judge_model (str): Model name for judging.

    Returns:
        Optional[str]: Generated response or None if failed.
    """
    if prompt_type == "Direct":
        return direct_answer(question, client, model)
    elif prompt_type == "COT":
        return chain_of_thought(question, client, model)
    elif prompt_type == "Self-Refine":
        return self_refine(question, client, model)
    elif prompt_type == "Self-Consistency":
        if judge_client is None:
            logger.error("Self-Consistency requires a judge_client.")
            return None
        return self_consistency(question, client, model, judge_client, judge_model)
    else:
        logger.error(f"Unknown prompt type: {prompt_type}")
        return None


def save_to_json(
        question: str,
        responses: list,
        output_file: str = "self_consistency_output.json"
) -> None:
    """
    Save the question, all responses, and selected responses to the specified file.

    Args:
        question (str): The problem to solve.
        responses (list): List of all generated responses.
        output_file (str): Path to the file where data will be saved.
    """
    try:
        output_data = {
            "question": question,
            "responses": responses,
        }
        with open(output_file, 'a') as f:
            json.dump(output_data, f, indent=4)
            f.write('\n')
        logger.debug(f"Data successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
