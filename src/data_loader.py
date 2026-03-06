import json
import random

def load_papers(train_path, dev_path):
    papers = []

    for path in [train_path, dev_path]:
        with open(path) as f:
            raw = json.load(f)

        for paper_id, paper in raw.items():
            sections = []
            for idx, sec in enumerate(paper.get("full_text", [])):
                paragraphs = [p.strip() for p in sec.get("paragraphs", []) if p is not None]
                sections.append({
                    "section_name": (sec.get("section_name") or "").strip(),
                    "section_idx": idx,
                    "text": " ".join(paragraphs),
                })

            qas = []
            for qa in paper.get("qas", []):
                answers = []
                for entry in qa.get("answers", []):
                    ans = entry.get("answer", {})

                    if ans.get("unanswerable"):
                        ans_type, ans_text = "unanswerable", "unanswerable"
                    elif ans.get("extractive_spans"):
                        ans_type = "extractive"
                        ans_text = " ".join(ans["extractive_spans"])
                    elif ans.get("free_form_answer"):
                        ans_type = "abstractive"
                        ans_text = ans["free_form_answer"].strip()
                    elif ans.get("yes_no") is not None:
                        ans_type = "yes_no"
                        ans_text = "yes" if ans["yes_no"] else "no"
                    else:
                        ans_type, ans_text = "unanswerable", "unanswerable"

                    answers.append({
                        "answer_type": ans_type,
                        "answer_text": ans_text,
                        "evidence": ans.get("evidence", []),
                        "highlighted_evidence": ans.get("highlighted_evidence", []),
                    })

                qas.append({
                    "question": qa["question"].strip(),
                    "question_id": qa["question_id"],
                    "answers": answers,
                })

            papers.append({
                "paper_id": paper_id,
                "title": paper.get("title", "").strip(),
                "abstract": paper.get("abstract", "").strip(),
                "sections": sections,
                "qas": qas,
            })

    return papers


def build_qa_records(papers):
    records = []
    for p in papers:
        for qa in p["qas"]:
            records.append({
                "paper_id": p["paper_id"],
                "question": qa["question"],
                "question_id": qa["question_id"],
                "gold_answers": [a["answer_text"] for a in qa["answers"]],
                "gold_answer_types": [a["answer_type"] for a in qa["answers"]],
                "gold_evidence": [a["evidence"] for a in qa["answers"]],
                "gold_highlighted": [a["highlighted_evidence"] for a in qa["answers"]],
            })
    return records


def sample_records(records, n=100, seed=42):
    # Filter out records where ALL annotators say unanswerable
    records = [
        r for r in records
        if not all(a == "unanswerable" for a in r["gold_answer_types"])
    ]

    by_paper = {}
    for r in records:
        by_paper.setdefault(r["paper_id"], []).append(r)

    rng = random.Random(seed)
    pool = [rng.choice(qs) for qs in by_paper.values()]
    rng.shuffle(pool)
    return pool[:n]

# For debugging/testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TRAIN_JSON, DEV_JSON
    print(sample_records(build_qa_records(load_papers(TRAIN_JSON,DEV_JSON))))
    # qa_records = build_qa_records(load_papers(TRAIN_JSON,DEV_JSON))
    # print(len(qa_records))
    # clean_records = [
    #     r for r in qa_records
    #     if not any("BIBREF" in ans or "TABREF" in ans
    #             for ans in r["gold_answers"])
    # ]
    # print(len(clean_records))
