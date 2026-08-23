import os
import sys
import json
import torch
from safetensors.torch import load_file

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer

import argparse

def main():
    parser = argparse.ArgumentParser(description="Chat with BellHartGPT in the terminal")
    parser.add_argument("--model_dir", type=str, default="models/nano-chat", help="Directory of the custom model")
    args = parser.parse_args()
    
    model_dir = args.model_dir
    
    if not os.path.exists(model_dir):
        print(f"Error: Could not find {model_dir}/")
        print("Please make sure you downloaded the fine-tuned model.")
        sys.exit(1)

    tokenizer_path = f"{model_dir}/tokenizer.json"
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)

    print(f"Loading config from {model_dir}/config.json...")
    with open(f"{model_dir}/config.json") as f:
        config_dict = json.load(f)
    
    config = GPTConfig(**{
        k: v for k, v in config_dict.items() 
        if hasattr(GPTConfig(), k)
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable fast inference on H100
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing model...")
    model = GPT(config).to(device)

    print(f"Loading safetensors weights from {model_dir}/model.safetensors...")
    state_dict = load_file(f"{model_dir}/model.safetensors")
    
    # Remove _orig_mod prefix if it exists (from torch.compile)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.eval()

    # We disable torch.compile for the terminal chat because compiling the 
    # dynamic autoregressive streaming loop takes minutes on Windows.
    # if hasattr(torch, "compile"):
    #     print("Compiling model for fast chat...")
    #     model = torch.compile(model, mode="default")

    print("\n" + "="*50)
    print(" 🤖 BellHart Chat is Ready!")
    print(" Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    chat_history = ""

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            # Format strictly like the training data
            chat_history += f"User: {user_input}\nAssistant: "
            
            # Tokenize and enforce memory limits
            tokens = tokenizer.encode(chat_history)
            
            # Simple Memory: If conversation gets too long, forget the oldest parts
            max_context = config.block_size - 512
            if len(tokens) > max_context:
                tokens = tokens[-max_context:]
                
            context = torch.tensor([tokens], dtype=torch.long, device=device)

            # Generate
            print("Assistant: ", end="", flush=True)
            response = ""
            
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                past_key_values = None
                idx = context
                
                for _ in range(512): # max_new_tokens
                    if past_key_values is not None:
                        past_len = past_key_values[0][0].size(-2)
                        if past_len >= config.block_size:
                            past_key_values = None
                            idx_cond = idx[:, -config.block_size:]
                        else:
                            idx_cond = idx[:, -1:]
                    else:
                        idx_cond = idx[:, -config.block_size:]

                    logits, past_key_values = model(idx_cond, past_key_values=past_key_values, use_cache=True)
                    logits = logits[:, -1, :] / 0.7 # temperature
                    
                    # Top-k
                    v, _ = torch.topk(logits, min(50, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    if idx_next.item() == tokenizer.eot_token:
                        break
                        
                    idx = torch.cat((idx, idx_next), dim=1)
                    
                    # Decode the single token and print it instantly
                    word = tokenizer.decode([idx_next.item()])
                    response += word
                    print(word, end="", flush=True)
                    
                    if "User:" in response or "\nUser" in response:
                        break
                        
            print() # Newline after generation completes
            
            # Clean up any partial hallucinations from the history
            if "User:" in response:
                response = response.split("User:")[0].strip()
            if "\nUser" in response:
                response = response.split("\nUser")[0].strip()

            # Append the actual response to history so it remembers context
            chat_history += response + "\n"

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
