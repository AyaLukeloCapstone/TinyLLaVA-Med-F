import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from pathlib import Path
from PIL import Image
import math
from transformers import set_seed, logging

# LLaVA model specific imports
from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

logging.set_verbosity_error()

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    set_seed(0)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

     # Lists to store metrics for each question
    inference_times = []
    tokens_per_question = []
    peak_memory_usages = []

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        #start measuring time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()

        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        total_tokens_generated = output_ids.size(1)  # Number of tokens generated for this question

        end_time.record()
        torch.cuda.synchronize()

        # Calculate inference time and memory usage for this question
        question_inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert milliseconds to seconds
        peak_memory_usage = torch.cuda.max_memory_allocated() / (1024**3)  # Convert bytes to GB
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats after each question to get the peak for each question

        inference_times.append(question_inference_time)
        tokens_per_question.append(total_tokens_generated)
        peak_memory_usages.append(peak_memory_usage)
        print()
        print(f"Question inference_time: {question_inference_time:.2f} seconds")
        print(f"Question Tokens per Second: {(total_tokens_generated/question_inference_time):.2f} tokens/sec")
        print(f"Question Peak Memory Usage (CUDA): {peak_memory_usage:.2f} GB")

        # Store the answer with the computed metrics
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx, "prompt": cur_prompt, "text": outputs, "answer_id": ans_id, "model_id": model_name, "metadata": {"inference_time": question_inference_time, "tokens_generated": total_tokens_generated, "memory_usage": peak_memory_usage}}) + "\n")
        ans_file.flush()

    
    # Output the average metrics after all questions have been processed
    average_inference_time = sum(inference_times) / len(inference_times)
    average_tokens_per_second = sum(tokens_per_question) / sum(inference_times)
    average_peak_memory_usage = sum(peak_memory_usages) / len(peak_memory_usages)

    
    print(f"Average Inference Time per Question: {average_inference_time:.2f} seconds")
    print(f"Average Tokens per Second: {average_tokens_per_second:.2f} tokens/sec")
    print(f"Average Peak Memory Usage (CUDA): {average_peak_memory_usage:.2f} GB")
    print("Model Memory footprint: ",model.get_memory_footprint() / (1024 ** 3))
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)


