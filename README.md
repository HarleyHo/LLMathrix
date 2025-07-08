# LLMathrix - A Testing Framework for Mathematical Ability of LLMs
LLMathrix is a testing framework for evaluating the mathematical ability of Large Language Models (LLMs). 
It uses a set of mathematical tasks and questions with different levels of difficulty to test the LLM's ability to perform complex mathematical calculations and solve problems.

## Methods
Rather than fine-tuning the LLM, in this framework we try to use a more easy approach to evaluate the LLM's ability to perform mathematical tasks, and that's **Prompt Engineering**. Each LLM are tested by three different methods, including chain of thought (CoT), self-refine, and self-consistency.

We choose two LLMs: [Qwen2.5-Math-1.5B](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen2.5-math-1.5b-instruct) and [DeepSeek-R1-Qwen-1.5B](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/text?currentTab=textChat&modelId=deepseek-r1-distill-qwen-1.5b).

The first model is fine-tuning with a large amount of mathematical tasks and questions, and the second model is a distilled model but for general questions. They both have a small number of parameters, so the mathematical ability of these models should be comparable and the effect the prompt engineering has on the model's performance should be significant.

## Datasets




## Usage
First you need to install the required packages:

```pip install -r requirements.txt```

Models are obtained from [Aliyun Bai Lian (百炼)](https://bailian.console.aliyun.com/?tab=home#/home) platform, so you need to register an account and get the **access api key** and **model base url**.

simply run the following command:

```python main.py```

