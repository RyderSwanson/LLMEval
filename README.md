# LLMEval

## Project Summary

### One-sentence description of the project
We're developing an automated evaluation framework to objectively compare the performance of multiple Large Language Models across key metrics like relevance, coherence, and factual accuracy, enabling enterprises to make informed decisions when selecting the right LLM for their specific needs.

### Additional information about the project
LLMEval is a comprehensive application designed to systematically evaluate and compare the performance of multiple Large Language Models (LLMs). This tool addresses the growing need for enterprises to make informed decisions when selecting LLMs for specific tasks.
Key Features:

1. Multi-dimensional Assessment: Evaluates LLMs on relevance, coherence, factual accuracy, creativity, and ethical considerations.
2. Advanced Metrics: Utilizes state-of-the-art NLP techniques for objective analysis.
3. Comparative Analysis: Provides consistent, repeatable evaluations across various prompts and domains.
4. Quantitative Insights: Employs statistical modeling to ensure robust, data-driven results.

Objective:
To develop a standardized, efficient framework that enables organizations to assess and compare LLM performances objectively, facilitating strategic decision-making in AI implementation.
This project aims to enhance the selection process of LLMs, ultimately improving the effectiveness and reliability of AI-driven solutions across various industries.

## Installation

### Prerequisites

Python 3.9 or higher
PiP

### Installation Steps

Open a terminal in the root directory (containing LLMEval.py).

Run the following command: ``` pip install -e . ``` (the -e makes this an editable package, omit if you do not plan to edit the source files)

The program can now be run from any terminal using the following syntax ``` python -m LLMEval [-h] --api_key API_KEY [--model MODEL] [--prompt PROMPT] ```

## Functionality

As it stands, the only functionality is the command line interface which allows the user to select a LLM to test against. 

In the near future once you have selected a model a series of tests will be kicked off and a model performance output will be generated.

## Known Problems

None so far.

## Contributing

1. Fork the project
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## Additional Documentation

TODO: Provide links to additional documentation that may exist in the repo, e.g.,
  * Sprint reports
  * User links

## License

TODO: Figure out what license to associate with this repo
