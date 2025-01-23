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

## Models 
### Models with supervised fine-tuning
| Model          | Size | Context | Train   | Link                                                       |
|:---------------|------|---------|---------|------------------------------------------------------------|
| 3D-RPE-LLaMA2-7B-Chat-HF  | 7B   | 16384   | Full FT | [Model](https://huggingface.co/xindian/3D-RPE-LLaMA2-7B-Chat-hf)

## Training
### Pre-trained weights
We use LLaMA2 models as the pre-trained weights and fine-tune them to long context window sizes. Download based on your choices.

| Pre-trained weights                                                        |
|:---------------------------------------------------------------------------|
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |

### Fine-tuning
'''
torchrun  fine-tune-cpe-chat2.py  \
        --model_name_or_path ./models/llama-2-7b-chat-hf \
        --bf16 True \
        --output_dir ./models/cpe-llama2-7b-chat-hf \
        --max_steps 3100    \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 16384  \
        --gradient_checkpointing True  \
        --pretraining_length 4096 \
        --lazy_preprocess False

'''
- There is no need to make supervised fine-tuning upon the fine-tuned context extended models. It is all right to directly use base model as Llama2-chat models, as the amount of long instruction following data is enough for SFT.
- Our long instruction following data can be found in [LongAlpaca-12k.json](https://huggingface.co/datasets/Yukang/LongAlpaca-12k).
  
### Get trainable weights in low-rank training
In low-rank training, we set embedding and normalization layers as trainable. Please use the following line to extract the trainable weights `trainable_params.bin` from `pytorch_model.bin`
```
python3 get_trainable_weights.py --checkpoint_path path_to_saving_checkpoints --trainable_params "embed,norm"
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin` and trainable parameters `trainable_params.bin`, save the resulting model into your desired path in the Hugging Face format:
```
python3 merge_lora_weights_and_save_hf_model.py \
        --base_model ./model/Llama-2-7b-hf \
        --peft_model path_to_saving_checkpoints \
        --context_size 8192 \
        --save_path path_to_saving_merged_model
```

## Evaluation 
```
python longbench-eval.py
'''
## Citation
If you find this project useful in your research, please consider citing:
This paper have been accepted by the conference AAAI-2025.

'''
@misc{ma20243drpeenhancinglongcontextmodeling,
      title={3D-RPE: Enhancing Long-Context Modeling Through 3D Rotary Position Encoding}, 
      author={Xindian Ma and Wenyuan Liu and Peng Zhang and Nan Xu},
      year={2024},
      eprint={2406.09897},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.09897}, 
}
```

