# RAG Local Setup on Apple Silicon using Zephyr 7B Beta
find the demo below
[![DEMO](https://github.com/taham655/RAG-Locally-with-zephyr7B/blob/c1a58a1c40e8a9e9a18c3ac23a0bffb7be9dfe07/YouTube-Logo.wine.svg)](https://www.youtube.com/watch?v=MxJLjSyVpxY)



This repository provides a comprehensive guide to running Retrieval-Augmented Generation (RAG) locally on Apple Silicon (I am running on macbook pro M1 8 gb ram), specifically using the Zephyr 7B Beta model, a fine-tuned version of Mistral 7B by the Hugging Face team.


## About Zephyr 7B Beta
![Zephyr7B-beta](https://github.com/taham655/RAG-Locally-with-zephyr7B/blob/8fd561e52824dd265fb8d43200d66cffe7f4eb05/zephyr.png)
Zephyr 7B Beta is a fine-tuned version of the Mistral 7B model, which has shown exceptional performance on MT-Bench, comparable to 13B chat models. The Hugging Face H4 team has fine-tuned Mistral 7B by removing the RLHF alignment layer and replacing it with DPO, resulting in Zephyr 7B Beta. This model not only outperforms Mistral 7B but also larger models like Vicuna 33B and WizardLM-30B in MMLU and MT-score benchmarks. Currently, Zephyr is a top-ranked 7B model on the Hugging Face leaderboard and competes with, or even surpasses, larger LLMs like Llama-2-70b, particularly in MT-bench performance.

### Benchmark
![BenchMark](https://github.com/taham655/RAG-Locally-with-zephyr7B/blob/d3170ccdd1c31a228469f70b5705438c072d5630/benchmarks.png)

### Model Download
The 5-bit quantized version of Zephyr 7B Beta can be found here: [Zephyr 7B Beta - TheBloke](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/blob/main/zephyr-7b-beta.Q5_K_S.gguf).


## Installation and Setup
Follow these steps to set up the environment and run the model:

### Cloning the Repository
`git clone https://github.com/taham655/RAG-Locally-with-zephyr7B.git`

### Downloading the Model
Download the model from [this link](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/blob/main/zephyr-7b-beta.Q5_K_S.gguf) and save it in the cloned directory.

### Installing Requirements
Install the necessary libraries using:
`pip install -r requirements.txt`

## Usage
After setting up, modify the `main.py` file to include the file you wish to perform RAG on. Execute the script, and you're all set to run Zephyr 7B Beta locally on your Apple Silicon.

![code](https://github.com/taham655/RAG-Locally-with-zephyr7B/blob/31c0d8687035a94169b2a5576c86b7ff805a4d64/code.png)

Hope you find it helpful :)
