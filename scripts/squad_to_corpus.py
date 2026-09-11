"""Convert a SQuAD 2.0 JSON file into a corpus ingest.py can load, plus a
held-out question set for benchmarking retrieval/verification.

Download SQuAD 2.0 yourself (not vendored here - it's ~4-40MB and public):

    curl -LO https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

Then convert and ingest:

    uv run python scripts/squad_to_corpus.py dev-v2.0.json data/squad
    uv run python ingest.py --dir data/squad

SQuAD 2.0 pairs each paragraph with questions that may be unanswerable from
it (`is_impossible: true`) - useful for checking that the answer/verify
pipeline says "unsupported" instead of guessing. This script writes one
.txt file per article (paragraphs joined in order) and a separate
questions.json (not ingested) with each question's text and whether SQuAD
considers it answerable, for comparing against HonestRAG's own verdict.
"""

import argparse
import json
import re
from pathlib import Path


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def convert(squad_path: Path, out_dir: Path) -> None:
    data = json.loads(squad_path.read_text())["data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = []
    for article in data:
        slug = _slugify(article["title"])
        paragraphs = [p["context"] for p in article["paragraphs"]]
        (out_dir / f"{slug}.txt").write_text("\n\n".join(paragraphs))

        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append(
                    {
                        "id": qa["id"],
                        "document": slug,
                        "question": qa["question"],
                        "is_impossible": qa["is_impossible"],
                        "answers": [a["text"] for a in qa["answers"]],
                    }
                )

    (out_dir / "questions.json").write_text(json.dumps(questions, indent=2))
    print(
        f"Wrote {len(data)} document(s) and {len(questions)} question(s) to {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("squad_json", type=Path, help="path to a SQuAD 2.0 JSON file")
    parser.add_argument("out_dir", type=Path, help="directory to write .txt files into")
    args = parser.parse_args()

    convert(args.squad_json, args.out_dir)


if __name__ == "__main__":
    main()
