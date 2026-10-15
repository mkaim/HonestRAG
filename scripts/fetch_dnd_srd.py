"""Fetch the D&D 5e SRD 5.2.1 eval corpus: markdown docs with real nested
headers, plus gold Q&A from the datapizza-ai-lab/dnd5e-srd-qa dataset on
Hugging Face (CC-BY-4.0). Not vendored here - see tests/eval/README.md for
why.

    uv run python scripts/fetch_dnd_srd.py tests/eval/data/dnd_srd

Writes <out_dir>/docs/*.md (20 files) and <out_dir>/gold.json, in the
generic gold format tests/eval/golden.py reads:

    [{"question": str, "answer": str, "tier": str,
      "passages": [{"content": str, "document": str}]}]

easy.json and medium.json (25 and 31 questions) are merged into one
gold.json, tagged by their source tier; `document` is one of the
docs/*.md filenames.
"""

import json
import urllib.request
from pathlib import Path

REPO = "datapizza-ai-lab/dnd5e-srd-qa"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
DOC_RANGES = [
    "001-018", "019-035", "036-046", "047-063", "064-076", "077-086",
    "087-103", "104-120", "121-137", "138-154", "155-175", "176-191",
    "192-203", "204-229", "230-252", "253-272", "273-292", "293-312",
    "313-332", "333-364",
]  # fmt: skip


def _fetch(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}/{path}") as response:
        return response.read()


def _doc_name(doc_range: str) -> str:
    return f"DND5eSRD_{doc_range}.md"


def fetch(out_dir: Path) -> None:
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for doc_range in DOC_RANGES:
        name = _doc_name(doc_range)
        (docs_dir / name).write_bytes(_fetch(f"dnd_srd_docs/{name}"))
    print(f"Wrote {len(DOC_RANGES)} document(s) to {docs_dir}")

    gold = []
    for tier in ("easy", "medium"):
        items = json.loads(_fetch(f"{tier}.json"))
        for item in items:
            gold.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "tier": tier,
                    "passages": [
                        {"content": p["content"], "document": p["document_path"]}
                        for p in item["passages"]
                    ],
                }
            )
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=2))
    print(f"Wrote {len(gold)} question(s) to {out_dir / 'gold.json'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="directory to write docs/ and gold.json into")
    args = parser.parse_args()

    fetch(args.out_dir)


if __name__ == "__main__":
    main()
