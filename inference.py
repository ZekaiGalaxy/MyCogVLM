# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--input_file", type=str, default="test.jsonl")
    parser.add_argument("--output_file", type=str, default="outputs.txt")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    with open(args.input_file) as reader:
        lines = reader.readlines()
        dataset = [json.loads(line) for line in lines]
    print(len(dataset))
    if len(dataset) == 0:
        print("NULL Folder!! Return!")
        return

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    # with open(args.input_file) as reader:
    #     lines = reader.readlines()
    #     dataset = [json.loads(line) for line in lines]
    outputs = []
    with open(args.output_file, "w+") as writer:
        with torch.no_grad():
           for data in tqdm(dataset):
               image_path = [data["file_path"]]
               query = f"Here is a 3D CAD img, please describe the image according to its Shape and Geometry, Dimensions, and Functional Features."
               if world_size > 1:
                   torch.distributed.broadcast_object_list(image_path, 0)
               image_path = image_path[0]
               
               response, _, _ = chat(
                   image_path, 
                   model, 
                   text_processor_infer,
                   image_processor,
                   query, 
                   history=None, 
                   image=None, 
                   max_length=args.max_length, 
                   top_p=args.top_p, 
                   temperature=args.temperature,
                   top_k=args.top_k,
                   invalid_slices=text_processor_infer.invalid_slices,
                   no_prompt=args.no_prompt
                   )
               outputs.append(response)
               writer.write(response + "\n")
               writer.write('#'*20 + "\n")
               writer.flush()


if __name__ == "__main__":
    main()
