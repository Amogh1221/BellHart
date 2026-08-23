import os
import sys
import json
import torch

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


def main():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = GPTConfig(**json.load(f))
    else:
        config = GPTConfig()
        print("No config.json found, using default config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device.type

    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)

    ckpt_path = "checkpoints/best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/latest.pt"

    if not os.path.exists(ckpt_path):
        import glob, re
        checkpoints = glob.glob("checkpoints/checkpoint-*.pt")
        valid_checkpoints = [f for f in checkpoints if re.search(r"checkpoint-(\d+)\.pt", os.path.basename(f))]
        if valid_checkpoints:
            ckpt_path = sorted(
                valid_checkpoints,
                key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", os.path.basename(x)).group(1))
            )[-1]
        
    if not os.path.exists(ckpt_path):
        print(f"No local checkpoint found. Downloading latest model from Hugging Face...")
        try:
            import re
            from huggingface_hub import HfApi, hf_hub_download
            os.makedirs("checkpoints", exist_ok=True)
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            files = api.list_repo_files(repo_id=config.hf_repo, repo_type="dataset")
            ckpt_files = [f for f in files if re.match(r"^checkpoints/checkpoint-\d+\.pt$", f)]
            if ckpt_files:
                ckpt_files.sort(
                    key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", x).group(1)),
                    reverse=True
                )
                target_file = ckpt_files[0]
            elif "checkpoints/latest.pt" in files:
                target_file = "checkpoints/latest.pt"
            else:
                target_file = "checkpoints/latest.pt"

            hf_hub_download(
                repo_id=config.hf_repo,
                filename=target_file,
                repo_type="dataset",
                local_dir="."
            )
            ckpt_path = target_file
            print(f"Download complete! ({target_file})")
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")
            sys.exit(1)

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from iteration {ckpt['iter_num']}")

    model.eval()

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if prompt:
        context = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    temperature = float(os.environ.get("TEMPERATURE", "0.8"))
    top_k = int(os.environ.get("TOP_K", "50"))
    top_p = float(os.environ.get("TOP_P", "0.95"))
    max_new = int(os.environ.get("MAX_NEW", "500"))

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    generated = tokenizer.decode(output[0].tolist())

    if prompt:
        print(generated[:len(prompt)] + "|" + generated[len(prompt):])
    else:
        print(generated)


if __name__ == "__main__":
    main()
