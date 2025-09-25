#!/usr/bin/env python3
import os, argparse, re
from pathlib import Path
import pandas as pd
from collections import defaultdict

def normalize_expr(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def check_concept(task_root: Path, concept: str):
    dataset_dir = task_root / concept / "dataset"
    splits = ["train", "val", "eval"]

    # 저장: norm -> {split: [row_idx,...]}, norm -> example_prompt
    occ = defaultdict(lambda: defaultdict(list))
    example_prompt = {}

    for split in splits:
        merged_path = dataset_dir / split / "merged.csv"
        if not merged_path.exists():
            print(f"[WARN] {concept}/{split}/merged.csv 없음, 건너뜀")
            continue
        df = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
        for idx, row in df.iterrows():
            norm = normalize_expr(row["prompt"])
            occ[norm][split].append(idx)
            if norm not in example_prompt:
                example_prompt[norm] = row["prompt"]

    # ---------- 1) 교차-스플릿 중복 (우선 출력) ----------
    cross = {}
    for norm, split_map in occ.items():
        present_splits = [s for s in splits if len(split_map.get(s, [])) > 0]
        if len(present_splits) > 1:
            cross[norm] = {s: split_map[s] for s in present_splits}

    if cross:
        print(f"[CROSS-SPLIT DUPLICATES] {concept}: {len(cross)} 개")
        for norm, split_map in cross.items():
            print(f"  Prompt: {example_prompt.get(norm, norm)!r}")
            for s in ["train","val","eval"]:
                if s in split_map:
                    rows = ", ".join(map(str, split_map[s]))
                    print(f"    - {s}: rows [{rows}]")
    else:
        print(f"[OK] {concept}: 교차-스플릿 중복 없음")

    # ---------- 2) 스플릿 내부 중복 (후순위로 별도 출력) ----------
    within_any = False
    for split in splits:
        merged_path = dataset_dir / split / "merged.csv"
        if not merged_path.exists():
            continue
        df = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
        # norm 기준으로 같은 split 내에서 2회 이상 등장
        df["__norm__"] = df["prompt"].map(normalize_expr)
        dup_groups = df.groupby("__norm__").indices  # dict: norm -> array of indices
        # indices는 dict-like이며, 각 값이 numpy array (row indices)
        split_dups = {norm: idxs for norm, idxs in dup_groups.items() if len(idxs) > 1}
        if split_dups:
            within_any = True
            print(f"[WITHIN-SPLIT DUPLICATES] {concept}/{split}: {len(split_dups)} 개")
            # 예시 출력
            for norm, idxs in split_dups.items():
                rows = ", ".join(map(str, idxs))
                # 예시 문자열 하나
                sample_prompt = df.iloc[idxs[0]]["prompt"]
                print(f"  Prompt: {sample_prompt!r} -> rows [{rows}]")
    if not within_any:
        print(f"[OK] {concept}: 각 split 내부 중복 없음")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--task", type=str, required=True, help="예: celebrity")
    args = parser.parse_args()

    task_root = Path(args.base_dir) / args.task
    if not task_root.exists():
        raise FileNotFoundError(f"Task root not found: {task_root}")

    concepts = sorted([p.name for p in task_root.iterdir() if (p / "dataset").exists()])
    for concept in concepts:
        check_concept(task_root, concept)

if __name__ == "__main__":
    main()
