"""Score NLI models on a Modal GPU, for models too slow on CPU.

Same scorer, cache files and resume behaviour as `cbench.nli`; only the forward pass
runs remotely. Needs `uv sync --group modal` and a token from `modal token new`;
`HF_TOKEN` in `.env` speeds up first downloads of model weights.

  uv run python -m cbench.nli_modal --models microsoft/deberta-xlarge-mnli --limit 512
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import modal

from cbench.candidates import Pair
from cbench.nli import NLI_MODELS, score_model

GPU_BATCH = 64
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.20.1", "sentencepiece==0.2.2", "protobuf==7.36.2",
                    "numpy==2.5.4", "scikit-learn==1.9.1", "pysbd==0.3.4")
    .add_local_python_source("cbench")
)
app = modal.App("cbench-nli", image=image)
weights = modal.Volume.from_name("cbench-hf-cache", create_if_missing=True)


def hf_token() -> str | None:
    if not modal.is_local():
        return os.environ.get("HF_TOKEN")
    from cbench.decide import load_env

    return load_env("HF_TOKEN")


# Defined the same way locally and in the container, which must agree on the function's dependencies.
secrets = [modal.Secret.from_dict({"HF_TOKEN": token} if (token := hf_token()) else {})]


@app.cls(gpu="T4", volumes={HF_CACHE: weights}, secrets=secrets, timeout=900, scaledown_window=60)
class RemoteScorer:
    model_name: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from cbench.nli import NLIScorer

        self.scorer = NLIScorer(self.model_name, GPU_BATCH, device="cuda")

    @modal.method()
    def score(self, pairs: list[Pair]) -> list[dict]:
        return self.scorer.score(pairs)


class ModalScorer:
    def __init__(self, model_name: str, gpu: str) -> None:
        self.remote = RemoteScorer.with_options(gpu=gpu)(model_name=model_name)

    def score(self, pairs: list[Pair]) -> list[dict]:
        return self.remote.score.remote(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=NLI_MODELS)
    parser.add_argument("--tests", nargs="+", default=["pair_test", "cand_k20"])
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--limit", type=int, help="score a seeded random sample of N pending pairs")
    args = parser.parse_args()

    def run(model_name: str) -> None:
        for test in args.tests:
            scored, seconds = score_model(model_name, test, limit=args.limit, batch_size=GPU_BATCH,
                                          make_scorer=lambda name, _: ModalScorer(name, args.gpu))
            if scored:
                print(f"{model_name} {test}: {scored} pairs in {seconds:.0f}s on {args.gpu}", flush=True)

    with modal.enable_output(), app.run(), ThreadPoolExecutor(len(args.models)) as pool:
        list(pool.map(run, args.models))


if __name__ == "__main__":
    main()
