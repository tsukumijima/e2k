"""
Exports the torch weights to numpy npy files.
"""

import argparse

import numpy as np
import torch
from safetensors.numpy import save_file

import train
from accent import AccentPredictor
from hp import EOS_IDX, SOS_IDX, ascii_entries, en_phones, kanas
from train import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--safetensors", action="store_true", help="Use safe tensors instead of numpy")
    parser.add_argument("--accent", action="store_true")
    parser.add_argument("--dim", type=int, default=None, help="Model dimension (auto-detected if not specified)")

    args = parser.parse_args()

    in_table = en_phones if args.p2k else ascii_entries
    out_table = kanas

    if not args.accent:
        # 次元数の自動検出
        if args.dim is None:
            state_dict = torch.load(args.model, map_location="cpu")
            if "e_emb.weight" in state_dict:
                dim = state_dict["e_emb.weight"].shape[1]
            elif "k_emb.weight" in state_dict:
                dim = state_dict["k_emb.weight"].shape[1]
            else:
                dim = 256
                print(f"Warning: Could not detect model dimension, using default: {dim}")
            print(f"Auto-detected model dimension: {dim}")
        else:
            dim = args.dim

        train.DIM = dim
        model = Model(p2k=args.p2k)
    else:
        model = AccentPredictor()

    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    if not args.fp32:
        model = model.half()

    weights = {}

    metadata = {}

    if not args.accent:
        metadata["in_table"] = "\0".join(in_table)
        metadata["out_table"] = "\0".join(out_table)
        metadata["sos_idx"] = str(SOS_IDX)  # safetensors only accepts str in metadata
        metadata["eos_idx"] = str(EOS_IDX)
    else:
        metadata["in_table"] = "\0".join(kanas[3:])

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            weights[name] = param.data.cpu().numpy()

    if args.safetensors:
        output = args.output if args.output.endswith(".safetensors") else f"{args.output}.safetensors"
        save_file(weights, output, metadata=metadata)
    else:
        weights["metadata"] = metadata
        output = args.output if args.output.endswith(".npz") else f"{args.output}.npz"
        np.savez(output, **weights)
