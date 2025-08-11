LLM Analysis Project
This repository contains code for analyzing and evaluating Large Language Models (LLMs). The project aims to investigate the impact of various text generation parameters such as temperature and max_new_tokens on the output of language models.

Features
Configure different text generation parameters
Compare model outputs with various settings
Evaluate text summarization quality
Analyze the effect of temperature on creativity and diversity in generated text
Usage
Prerequisites
python
pip install transformers torch
Sample Code
python
from transformers import GenerationConfig

# Define different configurations for testing
configurations = [
    {"name": "Default (max_new_tokens=50)", "params": {"max_new_tokens": 50}},
    {"name": "Very short output (max_new_tokens=10)", "params": {"max_new_tokens": 10}},
    {"name": "Low temperature (T=0.1)", "params": {"max_new_tokens": 50, "do_sample": True, "temperature": 0.1}},
    {"name": "Medium temperature (T=0.5)", "params": {"max_new_tokens": 50, "do_sample": True, "temperature": 0.5}},
    {"name": "High temperature (T=1.0)", "params": {"max_new_tokens": 50, "do_sample": True, "temperature": 1.0}}
]

# Run the model with each configuration and compare results
inputs = tokenizer(few_shot_prompt, return_tensors='pt')

print("=" * 80)
print("COMPARING DIFFERENT GENERATION CONFIGURATIONS")
print("=" * 80)

for config in configurations:
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            **config["params"]
        )[0],
        skip_special_tokens=True
    )
    
    print("\n" + "-" * 50)
    print(f'CONFIG: {config["name"]}')
    print("-" * 50)
    print(f'MODEL GENERATION:\n{output}')
    print("-" * 50)

print("\n" + "=" * 80)
print(f'BASELINE HUMAN SUMMARY:\n{summary}')
print("=" * 80)
Results
Analysis of the results shows:

The max_new_tokens parameter directly affects output length
Increasing temperature leads to more diverse outputs
Effective summarization requires careful prompt engineering and parameter tuning
Higher temperatures can produce more creative but potentially less coherent text
Key Parameters Explained
Temperature
Low (0.1): More deterministic, predictable outputs
Medium (0.5-0.7): Balance between creativity and coherence
High (1.0+): More creative, diverse, and sometimes unpredictable outputs
Max New Tokens
Controls the maximum length of generated text. Setting this too low can result in incomplete outputs.

Future Work
Implement more evaluation metrics
Test additional parameters like top_p and top_k
Compare performance across different model architectures
Develop better prompting techniques for summarization tasks
Contributing
Your suggestions and contributions to improve this project are highly appreciated. Please start collaborating by creating an Issue or Pull Request.

License
This project is licensed under the MIT License.






