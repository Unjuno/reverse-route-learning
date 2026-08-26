#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "roneneldan/TinyStories-8M"
DEFAULT_REVISION = "b5c14392fcdc61157a3cf4ab6944e9335e7ad6b3"
EXPECTED_SHA256 = "22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Download the exact TinyStories-8M checkpoint used by the v0.1 experiments")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--output-dir", type=Path, default=Path("models/TinyStories-8M"))
    p.add_argument("--skip-hash-check", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(
        hf_hub_download(
            repo_id=args.repo,
            filename="pytorch_model.bin",
            revision=args.revision,
        )
    )
    dst = args.output_dir / "pytorch_model.bin"
    dst.write_bytes(src.read_bytes())

    digest = sha256(dst)
    if not args.skip_hash_check and args.repo == DEFAULT_REPO and args.revision == DEFAULT_REVISION:
        if digest != EXPECTED_SHA256:
            dst.unlink(missing_ok=True)
            raise RuntimeError(f"checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {digest}")

    print(f"{dst}\nsha256={digest}\nrevision={args.revision}")


if __name__ == "__main__":
    main()
