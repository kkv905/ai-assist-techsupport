"""Generate a draft RAGAS test set; review it before replacing the golden JSON."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("tests/eval/golden_dataset_raw.csv"))
    parser.add_argument("--size", type=int, default=35)
    args = parser.parse_args()
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    docs = DirectoryLoader(str(args.data_dir), glob="**/*.md", loader_cls=TextLoader).load()
    generator = TestsetGenerator(llm=llm_factory("claude-sonnet-4-6", provider="anthropic"))
    generated = generator.generate_with_langchain_docs(docs, testset_size=args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_pandas().to_csv(args.output, index=False)
    print(f"Draft written to {args.output}. Manually review it before using it as golden_dataset.json.")


if __name__ == "__main__":
    main()
