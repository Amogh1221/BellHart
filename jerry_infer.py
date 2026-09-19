"""
jerry_infer.py  —  High-Speed INT4 Inference Engine for the Jerry Family
========================================================================
Supports:
  1. Instant loading and on-the-fly decompression of 4-bit packed weights (~55MB).
  2. Zero-Shot Sentiment Analysis (Positive, Neutral, Negative).
  3. Zero-Shot Multi-Class Intent & Topic Classification.
  4. General Natural Language Inference (Premise vs. Hypothesis).
  5. Interactive terminal testing interface.

Usage:
  python jerry_infer.py --sentiment "The camera quality on this phone is truly outstanding!"
  python jerry_infer.py --classify "Where is my package?" --candidates "shipping, billing, technical"
  python jerry_infer.py --interactive
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bert import ModernBertModel, BertConfig
from tokenizer import Tokenizer
from qat_int4 import unpack_weights_int4


class GeneralJerryModel:
    """Wrapper loading and executing INT4 packed General Purpose Jerry."""

    def __init__(self, checkpoint_path: str = "exported_models/GeneralJerry/general_jerry_int4.pt", device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer()

        print(f"Loading Jerry model from: {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        raw_config = ckpt.get("config", {})
        self.config = BertConfig(**raw_config) if isinstance(raw_config, dict) else raw_config
        self.group_size = ckpt.get("group_size", 64)
        self.classes = ckpt.get("classes", ["entailment", "neutral", "contradiction"])

        # Decompress 4-bit packed weights
        unpacked_state = {}
        for k, v in ckpt["packed_state_dict"].items():
            if k.endswith(".packed_int4"):
                base_name = k.replace(".packed_int4", "")
                scales = ckpt["packed_state_dict"][f"{base_name}.scales_fp16"]
                unpacked_state[base_name] = unpack_weights_int4(
                    v, scales, group_size=self.group_size, dtype=torch.float32
                )
            elif not k.endswith(".scales_fp16"):
                unpacked_state[k] = v.to(torch.float32)

        # Import classifier class
        from train_general_jerry import GeneralJerryClassifier
        self.model = GeneralJerryClassifier(self.config, num_classes=len(self.classes))
        self.model.load_state_dict(unpacked_state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"  [OK] Successfully loaded {ckpt.get('model_flavor', 'General Jerry')} on {self.device}.")

    def predict_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns softmax probabilities across [Entailment, Neutral, Contradiction]."""
        eot = self.tokenizer.eot_token
        tokens_p = self.tokenizer.encode(premise)
        tokens_h = self.tokenizer.encode(hypothesis)
        combined = tokens_p + [eot] + tokens_h + [eot]

        input_ids = torch.tensor([combined], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

        return {self.classes[i]: probs[i] for i in range(len(self.classes))}

    def predict_sentiment(self, text: str) -> Dict[str, any]:
        """
        Zero-Shot Sentiment Analysis using NLI:
          - Hypothesis: "This text expresses a positive sentiment."
          - High Entailment -> Positive
          - High Contradiction -> Negative
          - High Neutral -> Neutral
        """
        scores = self.predict_nli(premise=text, hypothesis="This text expresses a positive sentiment.")
        entail = scores.get("entailment", 0.0)
        contra = scores.get("contradiction", 0.0)
        neutral = scores.get("neutral", 0.0)

        if entail > contra and entail > neutral:
            sentiment = "Positive"
            conf = entail
        elif contra > entail and contra > neutral:
            sentiment = "Negative"
            conf = contra
        else:
            sentiment = "Neutral"
            conf = neutral

        return {
            "sentiment": sentiment,
            "confidence": conf,
            "raw_scores": {
                "positive": entail,
                "negative": contra,
                "neutral": neutral,
            },
        }

    def predict_classification(self, text: str, candidate_labels: List[str]) -> Dict[str, any]:
        """
        Zero-Shot Multi-Class Classification:
          Compares text against candidate hypotheses: "This text is about {label}."
        """
        entail_scores = []
        for label in candidate_labels:
            hypo = f"This text is about {label}."
            nli_res = self.predict_nli(premise=text, hypothesis=hypo)
            entail_scores.append(nli_res.get("entailment", 0.0))

        # Softmax over candidate entailment scores
        tensor_scores = torch.tensor(entail_scores)
        norm_probs = F.softmax(tensor_scores, dim=0).tolist()

        ranking = sorted(
            [{"label": candidate_labels[i], "score": norm_probs[i]} for i in range(len(candidate_labels))],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "top_label": ranking[0]["label"],
            "top_score": ranking[0]["score"],
            "all_scores": ranking,
        }


def main():
    parser = argparse.ArgumentParser(description="Jerry INT4 Inference & Zero-Shot Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="exported_models/GeneralJerry/general_jerry_int4.pt",
        help="Path to packed INT4 model checkpoint",
    )
    parser.add_argument("--sentiment", type=str, default="", help="Text to analyze for sentiment")
    parser.add_argument("--classify", type=str, default="", help="Text to classify")
    parser.add_argument("--candidates", type=str, default="support, sales, technical, billing", help="Comma-separated candidate labels")
    parser.add_argument("--premise", type=str, default="", help="Premise sentence for NLI")
    parser.add_argument("--hypothesis", type=str, default="", help="Hypothesis sentence for NLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive testing console")
    args = parser.parse_args()

    # Fallback to local model if available
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint '{model_path}' not found locally.")
        print("Please run `python train_general_jerry.py` first to create the INT4 model.")
        return

    engine = GeneralJerryModel(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    if args.sentiment:
        res = engine.predict_sentiment(args.sentiment)
        print(f"\nText: \"{args.sentiment}\"")
        print(f"Sentiment : {res['sentiment']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Details   : Positive: {res['raw_scores']['positive']:.3f} | Negative: {res['raw_scores']['negative']:.3f} | Neutral: {res['raw_scores']['neutral']:.3f}\n")
        return

    if args.classify:
        labels = [l.strip() for l in args.candidates.split(",")]
        res = engine.predict_classification(args.classify, labels)
        print(f"\nText: \"{args.classify}\"")
        print(f"Top Label: {res['top_label']} ({res['top_score']*100:.1f}%)\n")
        print("All Rankings:")
        for r in res["all_scores"]:
            print(f"  - {r['label']:<15}: {r['score']*100:.1f}%")
        print()
        return

    if args.premise and args.hypothesis:
        res = engine.predict_nli(args.premise, args.hypothesis)
        print(f"\nPremise   : \"{args.premise}\"")
        print(f"Hypothesis : \"{args.hypothesis}\"")
        for k, v in res.items():
            print(f"  - {k:<15}: {v*100:.1f}%")
        print()
        return

    if args.interactive:
        print("\n" + "=" * 60)
        print("  JERRY INT4 INTERACTIVE CONSOLE")
        print("  Type text to test Zero-Shot Sentiment & Intent Classification")
        print("  (Type 'exit' to quit)")
        print("=" * 60 + "\n")
        while True:
            try:
                user_text = input("Enter text > ").strip()
                if not user_text or user_text.lower() in ("exit", "quit"):
                    break
                s_res = engine.predict_sentiment(user_text)
                print(f"  -> Sentiment: {s_res['sentiment']} ({s_res['confidence']*100:.1f}%)")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
