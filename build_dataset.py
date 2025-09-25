#!/usr/bin/env python3
import argparse, csv, random, re
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------ Utils ------------------
def normalize_expr(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["expr","concept","type","extended_type","adv_type"]:
        if c not in df.columns:
            df[c] = ""
    return df[["expr","concept","type","extended_type","adv_type"]].copy()

def load_avail_csv(path: Path, source_tag: str, origin_class: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = ensure_cols(df)
    df["prompt"] = df["expr"]
    df["class"] = df["concept"]
    df["__norm_prompt__"] = df["prompt"].map(normalize_expr)
    df["__type_bucket__"] = source_tag  # short_adv / extended / extended_adv
    df["__origin_class__"] = origin_class
    return df

def split_targets_frac(total, weights=(1,1,2)):
    w = np.array(weights, dtype=float); w /= w.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base.tolist()  # [short, extended, ext_adv]

def split_targets_with_bias(total, weights=(1,1,2), priority=("extended_adv","extended","short_adv")):
    w = np.array(weights, dtype=float); w /= w.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    idx_map = {"short_adv":0, "extended":1, "extended_adv":2}
    for i in range(rem):
        base[idx_map[priority[i % len(priority)]]] += 1
    return base.tolist()

def proportional_quota(needs_map, total_unique, rng: random.Random):
    """needs_map: {'train':N, 'val':M, 'eval':K}"""
    splits = list(needs_map.keys())
    needs = np.array([needs_map[s] for s in splits], dtype=float)
    total_need = int(needs.sum())
    if total_need == 0 or total_unique == 0:
        return {s: 0 for s in splits}

    positive = [s for s in splits if needs_map[s] > 0]
    quotas = np.zeros(len(splits), dtype=int)

    min_floor = 1 if total_unique >= len(positive) else 0
    if min_floor == 1:
        for i, s in enumerate(splits):
            if needs_map[s] > 0:
                quotas[i] = 1
        total_unique -= len(positive)
        needs_adj = needs - (np.array([1 if needs_map[s] > 0 else 0 for s in splits]))
        needs_adj = np.clip(needs_adj, 0, None)
    else:
        needs_adj = needs.copy()

    if total_unique > 0 and needs_adj.sum() > 0:
        w = needs_adj / needs_adj.sum()
        raw = w * total_unique
        base = np.floor(raw).astype(int)
        rem = int(total_unique - base.sum())
        frac = raw - base
        order = np.argsort(-frac)
        for i in range(rem):
            base[order[i % len(order)]] += 1
        quotas += base

    for i, s in enumerate(splits):
        quotas[i] = int(min(quotas[i], needs_map[s]))
    return {s: int(quotas[i]) for i, s in enumerate(splits)}

def parse_ratio(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3 or any(p <= 0 for p in parts):
        raise ValueError("--ratio must be 'a,b,c' with positive ints")
    return tuple(parts)  # (short_adv, extended, extended_adv)

def parse_priority(s: str):
    parts = [x.strip() for x in s.split(",")]
    valid = {"short_adv","extended","extended_adv"}
    if len(parts) != 3 or set(parts) != valid:
        raise ValueError("--preserve_priority must be a permutation of short_adv,extended,extended_adv")
    return tuple(parts)

def assign_across_splits_unique_then_repeat(cands_df: pd.DataFrame,
                                            needs_map: dict,
                                            used_global: set,
                                            rng: random.Random):
    """
    1) used_global 제외한 고유 프롬프트 추출 → 셔플
    2) 고유 프롬프트를 split별 비례 배분(한 번씩만 사용)
    3) 부족분은 '해당 split의 고유로 배정된 것'에서만 복제
    """
    out = {s: [] for s in needs_map.keys()}
    if cands_df.empty:
        return out

    unique_df = cands_df.drop_duplicates(subset="__norm_prompt__")
    unique_df = unique_df[~unique_df["__norm_prompt__"].isin(used_global)].copy()

    recs = unique_df.to_dict("records")
    rng.shuffle(recs)

    total_unique = len(recs)
    quotas = proportional_quota(needs_map, total_unique, rng)

    idx = 0
    for s, q in quotas.items():
        if q <= 0: continue
        take = recs[idx: idx+q]
        idx += q
        out[s].extend(take)
        for r in take:
            used_global.add(r["__norm_prompt__"])

    for s, need in needs_map.items():
        cur = out[s]
        if len(cur) < need and len(cur) > 0:
            k = need - len(cur)
            reps = [rng.choice(cur) for _ in range(k)]
            out[s].extend(reps)
        # 고유가 0개면 규칙상 타 split 재사용 금지 → 비워둠(필요시 정책 변경 가능)
    return out

def finalize_rows(records, case_type):
    return [{
        "case_number": 0,
        "prompt": r["prompt"],
        "class": str(r["class"]).lower(),   # 항상 소문자
        "case_type": case_type,
    } for r in records]

# ------------------ Core Builder ------------------
def build_for_one_concept(task_root: Path, task_name: str, concept: str,
                          all_concepts: list, out_root: Path,
                          totals_by_split: dict, ratio_weights, preserve_priority,
                          seed: int = 42):
    rng = random.Random(seed + hash((task_name, concept)) % 1_000_000_007)

    ap_dir = task_root / concept / "avail_prompts"
    files = {
        "short_adv": ap_dir / "short_adv.csv",
        "extended": ap_dir / "extended.csv",
        "extended_adv": ap_dir / "extended_adv.csv",
    }
    for k,p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"[{task_name}/{concept}] missing file: {p}")

    # ERASE: 자기 것
    self_short = load_avail_csv(files["short_adv"], "short_adv", concept)
    self_ext   = load_avail_csv(files["extended"], "extended", concept)
    self_exta  = load_avail_csv(files["extended_adv"], "extended_adv", concept)
    self_all = pd.concat([self_short, self_ext, self_exta], ignore_index=True)

    # PRESERVE: 나머지 9명
    others = [c for c in all_concepts if c != concept]
    other_buckets = {"short_adv": [], "extended": [], "extended_adv": []}
    for oc in others:
        o_ap = task_root / oc / "avail_prompts"
        o_short = load_avail_csv(o_ap / "short_adv.csv", "short_adv", oc)
        o_ext   = load_avail_csv(o_ap / "extended.csv", "extended", oc)
        o_exta  = load_avail_csv(o_ap / "extended_adv.csv", "extended_adv", oc)
        other_buckets["short_adv"].append(o_short)
        other_buckets["extended"].append(o_ext)
        other_buckets["extended_adv"].append(o_exta)

    # 전역(해당 concept 전체) 중복 금지
    used_global = set()

    picked = {
        "erase": {"train": [], "val": [], "eval": []},
        "preserve": {"train": [], "val": [], "eval": []},
    }

    # ---- ERASE: 타입별 필요량 계산 후 '고유→복제' ----
    erase_targets = {}
    for split, total in totals_by_split.items():
        a,b,c = split_targets_frac(total, ratio_weights)   # 기본 1:3:6 (CLI 지정)
        erase_targets[split] = {"short_adv": a, "extended": b, "extended_adv": c}

    for tname in ["short_adv","extended","extended_adv"]:
        needs_map = {s: erase_targets[s][tname] for s in ["train","val","eval"]}
        pool = self_all[self_all["__type_bucket__"] == tname].sample(frac=1.0, random_state=rng.randint(0,1<<30))
        assigned = assign_across_splits_unique_then_repeat(pool, needs_map, used_global, rng)
        for s in ["train","val","eval"]:
            picked["erase"][s].extend(finalize_rows(assigned[s], "erase"))

    # ---- PRESERVE: 9명 균등 분배 + origin/타입별 '고유→복제' ----
    preserve_origin_plan = {}  # ← 초기화 필수
    for split, total in totals_by_split.items():
        base_per_origin = total // len(others)
        rem = total - base_per_origin * len(others)
        per_origin = [base_per_origin + (1 if i < rem else 0) for i in range(len(others))]

        per_origin_type_targets = []
        for k in per_origin:
            a,b,c = split_targets_with_bias(
                k, weights=ratio_weights,
                priority=preserve_priority  # 기본: ext_adv → extended → short_adv
            )
            per_origin_type_targets.append({"short_adv": a, "extended": b, "extended_adv": c})
        preserve_origin_plan[split] = per_origin_type_targets

    # origin별/타입별 고유→복제
    for idx, oc in enumerate(others):
        for tname in ["short_adv","extended","extended_adv"]:
            pools = {
                "short_adv": other_buckets["short_adv"][idx],
                "extended":  other_buckets["extended"][idx],
                "extended_adv": other_buckets["extended_adv"][idx],
            }
            needs_map = {
                "train": preserve_origin_plan["train"][idx][tname],
                "val":   preserve_origin_plan["val"][idx][tname],
                "eval":  preserve_origin_plan["eval"][idx][tname],
            }
            pool = pools[tname].sample(frac=1.0, random_state=rng.randint(0,1<<30))
            assigned = assign_across_splits_unique_then_repeat(pool, needs_map, used_global, rng)
            for s in ["train","val","eval"]:
                picked["preserve"][s].extend(finalize_rows(assigned[s], "preserve"))

    # ---- 저장 ----
    out_dir = out_root / task_name / concept / "dataset"
    for sub in ["train","val","eval"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for split in ["train","val","eval"]:
        erase_df = pd.DataFrame(picked["erase"][split], columns=["case_number","prompt","class","case_type"])
        preserve_df = pd.DataFrame(picked["preserve"][split], columns=["case_number","prompt","class","case_type"])
        erase_df.to_csv(out_dir / split / "erase.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        preserve_df.to_csv(out_dir / split / "preserve.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        merged_df = pd.concat([erase_df, preserve_df], ignore_index=True)
        merged_df.to_csv(out_dir / split / "merged.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    return {
        "concept": concept,
        "train": {"erase": len(picked["erase"]["train"]), "preserve": len(picked["preserve"]["train"])},
        "val":   {"erase": len(picked["erase"]["val"]),   "preserve": len(picked["preserve"]["val"])},
        "eval":  {"erase": len(picked["erase"]["eval"]),  "preserve": len(picked["preserve"]["eval"])},
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--train_n", type=int, default=100)
    parser.add_argument("--val_n", type=int, default=50)
    parser.add_argument("--eval_n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=str, default="1,3,6",
                        help="short_adv,extended,extended_adv ratio. e.g., '1,3,6'")
    parser.add_argument("--preserve_priority", type=str, default="extended_adv,extended,short_adv",
                        help="Remainder priority for preserve per-origin allocation (permute of the three types)")
    args = parser.parse_args()

    ratio_weights = parse_ratio(args.ratio)
    preserve_priority = parse_priority(args.preserve_priority)

    base_dir = Path(args.base_dir)
    task_root = base_dir / args.task
    if not task_root.exists():
        raise FileNotFoundError(f"Task root not found: {task_root}")

    concepts = sorted([p.name for p in task_root.iterdir() if (p / "avail_prompts").exists()])
    if len(concepts) < 2:
        raise ValueError(f"Need >=2 concepts under {task_root}")

    totals_by_split = {"train": args.train_n, "val": args.val_n, "eval": args.eval_n}
    for concept in concepts:
        rep = build_for_one_concept(
            task_root, args.task, concept, concepts, base_dir,
            totals_by_split, ratio_weights, preserve_priority, args.seed
        )
        print(f"[DONE] {rep}")

if __name__ == "__main__":
    main()
