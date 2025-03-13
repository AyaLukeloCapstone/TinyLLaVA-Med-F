# TinyLLaVA-Med-F

<p align="center">
    <img src="images/methodology-fig-capstone.png" width="100%" height="200%"> <br>
 
  *Overview of the methodology framework across four key stages for adapting MLLMs to resource-limited healthcare settings. Starting with the Optimization phase, the general- purpose TinyLLaVA model undergoes fine-tuning and quantization into variants TinyLLaVA-Med-F, FQ4, FQ8 while the quantization of LLaVA-Med leads to variants LLaVA-Med-Q4 and Q8. The Evaluation phase tests the models on benchmark datasets (VQA, SLAKE, PathVQA) and with GPT-4. In the Deployment stage, models are implemented on consumer devices to assess memory usage. Finally, the Integration into Hospital Systems stage explores their integration into healthcare systems for improved radiology services.*
</p>

## Table of Contents

- [Abstract](#Abstract)
- [Data Download](#data-download)
- [TinyLLaVA-Med](#tinyllava-med)
  - [Requirements and Installation](#requirements-and-installation)
  - [TinyLLaVA Models](#tinyllava-models)
  - [Demo](#demo)
  - [Train](#train)
  - [TinyLLaVA-Med Evaluation](#tinyllava-med-evaluation)

## Abstract

The critical shortage of medical professionals in low- resource countries, notably in Africa, hinders adequate health- care delivery. AI, particularly Multimodal Large Language Mod- els (MLLMs), can enhance the efficiency of healthcare systems by assisting in medical image analysis and diagnosis. However, the deployment of state-of-the-art MLLMs is limited in these regions due to the high computational demands that exceed the capabil- ities of consumer-grade GPUs. This paper presents a framework for optimizing MLLMs for resource-constrained environments. We introduce optimized medical MLLMs including TinyLLaVA- Med-F, a medical fine-tuned MLLM, and quantized variants (TinyLLaVA-Med-FQ4, TinyLLaVA-Med-FQ8, LLaVA-Med-Q4, and LLaVA-Med-Q8) that demonstrate substantial reductions in memory usage without significant loss in accuracy. Specifically, TinyLLaVA-Med-FQ4 achieves the greatest reductions, lowering dynamic memory by approximately 89% and static memory by 90% compared to LLaVA-Med. Similarly, LLaVA-Med-Q4 reduces dynamic memory by 65% and static memory by 67% compared to state-of-the-art LLaVA-Med. These memory reduc- tions make these models feasible for deployment on consumer- grade GPUs such as RTX 3050. This research underscores the potential for deploying optimized MLLMs in low-resource settings, providing a foundation for future developments in accessible AI-driven healthcare solutions.

## Data Download

| Alignment data files                                                                                                            |       Size |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_alignment_500k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/alignment/llava_med_alignment_500k.json) | 341.52 MiB |

| Instruction-Tuning data files                                                                                                                            |       Size |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_instruct_10k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_10k.json)                               |  19.24 MiB |
| [llava_med_instruct_60k.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k.json)                               |  84.65 MiB |
| [llava_med_instruct_60k_inline_mention.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json) |  83.61 MiB |
| [llava_med_instruct_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_fig_captions.json)             | 161.39 MiB |

| Evaluation files                                                                                                                                                                               |       Size |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_eval_qa50_qa.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl)                                                                       | 256.18 KiB |
| [llava_med_eval_qa50_fig_captions.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_fig_captions.json)                                                     |  51.82 KiB |
| [llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json](https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json) | 100.97 KiB |

| Image URLS                                                                                                      |       Size |
| --------------------------------------------------------------------------------------------------------------- | ---------: |
| [llava_med_image_urls.jsonl](https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl) | 122.82 MiB |

[download_images.py](llava/data/download_images.py) is used to download the PMC articles using the above image_urls file and extract the images

To download our langauge-image multimodal instruction-folllowing dataset, please run the following script:

```bash
sh download_data.sh
```




# TinyLLaVA-Med

## Requirements and Installation

We recommend the requirements as follows.

1. Clone this repository and navigate to LLaVA folder

```bash
cd TinyLLaVA-Med-F
```

2. Install Package

```Shell
conda create -n tinyllava-f python=3.10 -y
conda activate tinyllava-f
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases

```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## TinyLLaVA Models 

### 8-bit Quantized Variants
#### TinyLLaVA-Med-FQ4
#### LLaVA-Med-Q4

### 4-bit Quantized Variants
#### TinyLLaVA-Med-FQ8 

#### LLaVA-Med-Q8


### Pretrained Models


## Demo

### Gradio Web Demo

Launch a local web demo by running:

```shell
python tinyllava/serve/app.py --model-path bczhou/TinyLLaVA-3.1B --model-name TinyLLaVA-3.1B
```

## Train

### Stage 1: Extensive Finetuning

#### Biomedical Alignment
```Shell
DATA_PATH= /path/to/llava_med_alignment_500k.json \
IMAGE_PATH= /path/to/your-image-folder

LLM_VERSION=bczhou/TinyLLaVA-1.5B
VT_VERSION=bczhou/TinyLLaVA-1.5B-SigLIP

output_directory= /path/to/biomedical-alignment/checkpoints
wandb_path= Tinyllava_biomedical-alignment

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path
```

#### Instruction Tuning

```Shell
DATA_PATH= /path/to/llava_med_instruct_60k_inline_mention.jsonl \
IMAGE_PATH= /path/to/your-image-folder

LLM_VERSION= /path/to/biomedical-alignment/checkpoints
VT_VERSION=bczhou/TinyLLaVA-1.5B-SigLIP

output_directory= /path/to/instruction-tuning/checkpoints
wandb_path= Tinyllava_instruction-tuning

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path
```

#### Downstream Finetuning
```Shell

LLM_VERSION=/path/to/instruction-tuning/checkpoints
VT_VERSION=bczhou/TinyLLaVA-1.5B-SigLIP


DATA_PATH=/path/to/3vqa/train_all.json
IMAGE_PATH=/path/to/3vqa/images

output_directory=/path/to/downstream-finetuning/checkpoints 
wandb_path=Tinyllava_downstream-finetuning

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 18 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path

```

### Stage 2: Post - Training Quantization


## TinyLLaVA-Med Evaluation

### 1. Medical VQA

Three Medical VQA datasets are considered in our experiments, including VQA-Rad, SLAKE, Pathology-VQA. We use VQA-Rad as the running example to illustrate how LLaVA-Med is applied to a downstream scenario.

#### - Prepare Data

1. Please see VQA-Rad [repo](https://paperswithcode.com/dataset/vqa-rad) for setting up the dataset.
2. Generate VQA-Rad dataset for TinyLLaVA-Med conversation-style format (the same format with instruct tuning). For each dataset, we process it into three components: `train.json`, `test.json`, `images`.

#### - Fine-tuning

<details>
<summary> Detailed script to fine-tune to downstream datasets: TinyLLaVA-Med-1.5B. </summary>

```Shell
DATA_PATH= /path/to/your-VQA-train-json-file \
IMAGE_PATH= /path/to/your-VQA-image-folder

output_directory= path to the checkpoints output folder
wandb_path= Tinyllava_SIGLIP_SG3_EX3_VQARAD

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model True \
    --bf16 True \
    --output_dir $output_directory \
    --num_train_epochs 18 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $wandb_path
```

</details>

#### - Evaluation

(a) Generate TinyLLaVA-Med responses on ScienceQA dataset

```Shell
python tinyllava/eval/model_vqa_med.py --model-name /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --question-file path/to/eval/vqa_rad/test.json \
    --image-folder path/to/eval/vqa_rad/images \
    --answers-file /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

(b) Evaluate the generated responses

```Shell
python ../llava/eval/run_eval.py \
    --gt /path/to/eval/vqa_rad/test.json \
    --pred /path/to/checkpoint_tinyllava/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```



### 2. GPT 


## Acknowledgement

- Our project is built upon [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) and [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). They provided us the code, base models, and dataset with the amazing multimodal and langauge capabilities!
