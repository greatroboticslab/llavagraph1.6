import torch
from transformers import pipeline
import argparse
import json
import os
import re

def main(args):
    print(f"Loading Llama model...")
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Handle the input file argument
    if not os.path.exists(args.conversation_file):
        print(f"Error: {args.conversation_file} not found.")
        return

    with open(args.conversation_file, "r") as f:
        data = json.load(f)

    if args.subset:
        data = data[:args.subset]

    results = []
    correct_count = 0
    label_map = {"A": "noise", "B": "sine", "C": "square"}

    for entry in data:
        image_path = entry.get("image", "").lower()
        
        # Ground Truth Extraction
        if "noise" in image_path: ground_truth = "noise"
        elif "sine" in image_path: ground_truth = "sine"
        elif "square" in image_path: ground_truth = "square"
        else: ground_truth = "unknown"

        # Extract full descriptive sentences from the LLaVA conversation
        # This captures the 'unique features' rather than just the label
        full_description = ""
        conv_key = "conversation" if "conversation" in entry else "conversations"
        for turn in entry.get(conv_key, []):
            ans = turn.get("answer") or turn.get("value")
            if ans:
                full_description += f"{ans.replace('<s>', '').strip()} "

        # --- THE CORE FEATURE-DETECTION LOGIC ---
        system_content = (
            "You are a specialized engineer analyzing waveform descriptions. "
            "IMPORTANT: The vision model often mislabels signals as 'square wave-like'. "
            "You must prioritize GEOMETRIC FEATURES over labels.\n\n"
            "CLASSIFICATION LOGIC:\n"
            "1. RANDOM NOISE (A): Look for descriptions of 'sawtooth-like appearance', 'non-uniform density', "
            "'clusters of drops', or 'random/irregular glitches'. If the drops occur in clusters or are inconsistent, it is NOISE.\n"
            "2. SINE WAVE (B): Look for descriptions of a 'stepped or stair-like pattern', 'oscillations', "
            "'regular spacing of peaks and troughs', or 'sequential vertical transitions'. If it oscillates consistently "
            "but is called 'square-like', it is actually a SINE wave.\n"
            "3. SQUARE WAVE (C): Look for 'two distinct displacement levels', 'horizontal sections of stability', "
            "or 'binary states'. It must stay horizontal/flat for a clear duration to be a SQUARE wave."
        )

        user_prompt = f"Waveform Description:\n{full_description}\n\nBased on the geometric rules, is this A, B, or C? Answer format: 'Result: [Letter]'"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]

        outputs = pipe(messages, max_new_tokens=150, do_sample=False)
        model_response = outputs[0]["generated_text"][-1]["content"]

        # Extraction and Scoring
        match = re.search(r"Result:\s*([A-C])", model_response, re.IGNORECASE)
        predicted_letter = match.group(1).upper() if match else "A"
        predicted_label = label_map.get(predicted_letter)

        is_correct = (predicted_label == ground_truth)
        if is_correct: correct_count += 1

        print(f"Image: {os.path.basename(image_path)[:20]}... | GT: {ground_truth:6} | Pred: {predicted_label:6} | {'✓' if is_correct else '✗'}")

        results.append({
            "image": image_path,
            "gt": ground_truth,
            "pred": predicted_label,
            "is_correct": is_correct,
            "reasoning": model_response
        })

    accuracy = (correct_count / len(data)) * 100 if len(data) > 0 else 0
    print(f"\nFinal Accuracy for {os.path.basename(args.conversation_file)}: {accuracy:.2f}%")
    
    with open(args.output_file, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conversation-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--subset", type=int)
    args = parser.parse_args()
    main(args)
