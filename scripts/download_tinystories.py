#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="roneneldan/TinyStories-8M")
    p.add_argument("--output-dir", type=Path, default=Path("models/TinyStories-8M"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=args.repo, filename="pytorch_model.bin")
    dst = args.output_dir / "pytorch_model.bin"
    dst.write_bytes(Path(src).read_bytes())
    print(dst)


if __name__ == "__main__":
    main()
