import argparse
import torch
import os
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer, CLIPImageProcessor # Added CLIPImageProcessor

import warnings
warnings.filterwarnings("ignore")

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # --- FIX: Ensure image_processor is not None ---
    if image_processor is None:
        print("[INFO] image_processor not found in model path. Loading default CLIP processor...")
        # LLaVA 1.5 usually uses this specific CLIP model
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    # -----------------------------------------------

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # Filter to ensure we only try to open valid image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    images = [f for f in os.listdir(args.image_folder) if f.lower().endswith(valid_extensions)]
    
    if (args.subset):
        images = images[:args.subset]

    results = []
    for testImage in images:
        conv = conv_templates[args.conv_mode].copy()
        print(f"Evaluating {testImage} ...")
        
        image_path = os.path.join(args.image_folder, testImage)
        image = load_image(image_path)
        image_size = image.size
        
        # image_processor is now guaranteed to be valid
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        questions = [
            "Describe the repetition and density of the signal across the timeline.", 
            "Identify the shape of the most prominent 'edges' or 'transitions'.", 
            "State the vertical range (min/max) and the polarity (positive/negative) of the displacement."
        ]
        
        result = {
            "image": testImage,
            "conversation": []
        }
        
        # Keep a copy for multi-turn if needed, but here we reset image per turn
        current_image_for_conv = image 

        for question in questions:
            inp = question

            if current_image_for_conv is not None:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                current_image_for_conv = None # Ensure image token is only added once per image
            
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            # Clean up potential </s> or other end tokens from the string
            if outputs.endswith("</s>"):
                outputs = outputs[:-4]
                
            conv.messages[-1][-1] = outputs
            result["conversation"].append({"question": question, "answer": outputs})
            
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
                
        results.append(result)

    # Ensure result directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w") as outputFile:
        json.dump(results, outputFile, indent=2)
    print(f"Evaluation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--subset", type=int, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2) # Lowered for more consistent eval
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
