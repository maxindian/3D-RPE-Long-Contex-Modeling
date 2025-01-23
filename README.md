# 3D-RPE-Long-Contex-Modeling

<p align="center" width="100%">
<img src=image/3D-RPE.png alt="TJU-QNLU" style="width: 100%; min-width: 300px; display: block; margin:auto;">
</p>

[![Huggingface Models](https://img.shields.io/badge/Models-Huggingface%20Models-bron)](https://huggingface.co/xindian/3D-RPE-LLaMA2-7B-Chat-hf/)
[![Data](https://img.shields.io/badge/Data-LongAlpaca%2012k-light)](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
[![Paper](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/pdf/2406.09897)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/dvlab-research/LongLoRA/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-orange.svg)](https://github.com/dvlab-research/LongLoRA/blob/main/DATA_LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](https://github.com/dvlab-research/LongLoRA/blob/main/WEIGHT_LICENSE)

## TABLE OF CONTENTS
1. [Requirements](#usage-requirements)
2. [Installation and quick guide](#installation-and-quick-guide)
3. [Data](#longalpaca-data)
4. [Models](#models)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Citation](#citation)

## Usage Requirements
To download and use the [pre-trained weights](#pre-trained-weights) you will need:
1. Hugging Face (HF) account with valid email. Note, the email used for HF must alse be used for the license agreement.
2. Accept the Meta [license and acceptable use policy](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) 

## Installation and Quick Guide
To install and run the application:
1. [Fork this repo](https://github.com/maxindian/3D-RPE-Long-Contex-Modeling) on github
2. Clone the repository on your local machine, using git clone and pasting the url of this project.
3. Run the following code:
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## LongAlpaca Data

LongAlpaca-12k contains 9k long QA data that we collected and 3k short QA sampled from the original [Alpaca data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). This is to avoid the case that the model might degrade at short instruction following. The data we collect contains various types and amounts as the following figure.

| Data           | Short QA | Long QA  | Total    | Download |
|:---------------|----------|----------|----------|----------|
| LongAlpaca-12k | 3k       | 9k       | 12k      | [Link](https://huggingface.co/datasets/Yukang/LongAlpaca-12k) |

Following the original Alpaca format, our Long QA data uses the following prompts for fine-tuning:
- `instruction`: `str`, describes the task the model should perform. For example, to answer a question after reading a book section or paper. We vary the contents and questions to make instructions diverse.
- `output`: `str`, the answer to the instruction.
We did not use the `input` format in the Alpaca format for simplicity.



