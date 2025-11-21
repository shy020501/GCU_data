import os
import gc
import shutil
import argparse
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from vlm import VLM
from sd import SD

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BASE_FORMATS = {
    'celebrity': ["Photo of ", "Image of ", "Portrait of ", "Close-up shot of ", "Realistic rendering of "],
    'style': ["A painting by ", "Art by ", "Artwork by ", "Picture by ", "Style of "]
}

# 부족한 경우 기록용 전역 로그
SHORTAGE_LOG = []

# ======================= Utils =======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def eval_with_backoff(eval_fn, paths, concept, min_bs=1):
    """OOM 등 오류 시 배치를 줄여가며 끝까지 평가."""
    if not paths:
        return []
    bs = len(paths)
    start = 0
    preds_all = []
    while start < len(paths):
        end = min(len(paths), start + bs)
        sub = paths[start:end]
        try:
            preds_all.extend(eval_fn(sub, concept))
            start = end
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > min_bs:
                torch.cuda.empty_cache()
                bs = max(min_bs, bs // 2)
                print(f"[Backoff] OOM detected. Reducing eval batch to {bs}")
            else:
                print(f"[Consume Warning] eval failed: {e}")
                preds_all.extend([0] * len(sub))
                start = end
    return preds_all

def move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir):
    """선택된 이미지만 최종 폴더로 이동 (기존 파일은 건드리지 않음)."""
    ensure_dir(final_image_dir)
    selected_abs = set(os.path.abspath(p) for p in selected_paths)
    for src in selected_abs:
        try:
            dst = os.path.join(final_image_dir, os.path.basename(src))
            shutil.move(src, dst)
        except Exception as e:
            print(f"[Warning] Failed to move {os.path.basename(src)}: {e}")

def pick_devices():
    """사용 가능한 디바이스 리스트를 문자열로 반환."""
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    elif n == 1:
        return ["cuda:0"]
    else:
        return ["cuda:0", "cuda:1"]

def load_existing_avail(avail_dir, filename, columns):
    """이미 존재하는 avail_prompts CSV를 불러오고 expr 집합을 반환."""
    ensure_dir(avail_dir)
    path = os.path.join(avail_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "expr" not in df.columns:
            raise ValueError(f"[Error] {path} must contain 'expr' column.")
        return df, set(df["expr"].astype(str).tolist()), path
    else:
        df = pd.DataFrame(columns=columns)
        return df, set(), path

# ======================= Phase context managers =======================

class SDGroup:
    """with 블록 안에서만 SD 파이프라인을 로드해 GPU에 상주시키고, 나올 때 메모리 해제."""
    def __init__(self, devices):
        self.device_strs = devices
        self.pipes = None
    def __enter__(self):
        self.pipes = [SD(torch.device(d)) for d in self.device_strs]
        return self.pipes
    def __exit__(self, exc_type, exc, tb):
        for p in self.pipes or []:
            del p
        self.pipes = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class VLMGroup:
    """
    with 블록 안에서만 VLM을 로드해 GPU에 상주시키고, 나올 때 메모리 해제.
    ※ 여기서는 device_map='auto'로 단일 인스턴스를 생성하여 가용 GPU들에 샤딩합니다.
    """
    def __init__(self, model_id, devices):
        self.model_id = model_id
        self.devices = devices  # 디버그용 정보
        self.pipes = None
    def __enter__(self):
        # 단일 인스턴스 + device_map="auto" → CUDA_VISIBLE_DEVICES에 노출된 GPU에 자동 샤딩
        self.pipes = [VLM(self.model_id, device=None, device_map=None)]
        print(f"[VLMGroup] Loaded single VLM with device_map='auto' over devices: {self.devices}")
        return self.pipes
    def __exit__(self, exc_type, exc, tb):
        for p in self.pipes or []:
            del p
        self.pipes = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ======================= Per-phase parallel helpers =======================

def _gen_on_device(sd_pipeline: SD, pairs, save_dir, seed):
    """단일 디바이스에서 [(idx, prompt)] 배치를 생성 → [(idx, img_path, prompt)] 리턴"""
    if not pairs:
        return []
    idxs, prompts = zip(*pairs)
    paths = sd_pipeline.create_image(list(prompts), save_dir=save_dir, seed=seed)
    return list(zip(idxs, paths, prompts))

def _eval_on_device(vlm_pipeline, idx_path_pairs, concept, eval_batch):
    """단일 디바이스에서 [(idx, path)] 평가 → [(idx, pred)] 리턴 (배치 단위 backoff 포함)."""
    if not idx_path_pairs:
        return []
    idxs, paths = zip(*idx_path_pairs)
    preds = []
    for i in range(0, len(paths), eval_batch):
        batch_paths = list(paths[i:i+eval_batch])
        preds.extend(eval_with_backoff(vlm_pipeline.eval_image, batch_paths, concept))
    return list(zip(idxs, preds))

def parallel_generate_all(sd_pipes, prompts, sd_batch, tmp_image_dir, seed):
    """
    SD 파이프(여러 디바이스)를 사용해 '생성 단계'를 디바이스별 병렬로 수행.
    순서를 보존한 [(img_path, prompt)] 리스트를 반환.
    """
    ensure_dir(tmp_image_dir)
    if not prompts:
        return []

    indexed = list(enumerate(prompts))
    num_dev = max(1, len(sd_pipes))
    stride = sd_batch * num_dev
    results = []

    for start in range(0, len(indexed), stride):
        slice_pairs = indexed[start:start+stride]
        shards = []
        for d in range(num_dev):
            shard = slice_pairs[d*sd_batch:(d+1)*sd_batch]
            shards.append(shard)

        futures = []
        with ThreadPoolExecutor(max_workers=num_dev) as pool:
            for pipe, shard in zip(sd_pipes, shards):
                futures.append(pool.submit(_gen_on_device, pipe, shard, tmp_image_dir, seed))
            for fut in as_completed(futures):
                results.extend(fut.result())

    results.sort(key=lambda x: x[0])  # (idx, path, prompt)
    return [(path, prompt) for _, path, prompt in results]

def parallel_eval_all(vlm_pipes, pairs, concept, eval_batch):
    """
    VLM 파이프(여러 디바이스)를 사용해 '평가 단계' 병렬 수행.
    device_map='auto'에서는 보통 파이프가 1개이며 내부적으로 멀티GPU를 씁니다.
    """
    if not pairs:
        return []

    concept = concept.replace('_', ' ')
    indexed = list(enumerate(pairs))
    num_pipes = max(1, len(vlm_pipes))  # auto면 1

    results = []
    chunk_size = max(1, (len(indexed) + num_pipes - 1) // num_pipes)
    chunks = [indexed[i:i+chunk_size] for i in range(0, len(indexed), chunk_size)]
    chunks += [[]] * (num_pipes - len(chunks))

    futures = []
    with ThreadPoolExecutor(max_workers=num_pipes) as pool:
        for pipe, chunk in zip(vlm_pipes, chunks):
            idx_path_pairs = [(idx, pair[0]) for idx, pair in chunk]
            futures.append(pool.submit(_eval_on_device, pipe, idx_path_pairs, concept, eval_batch))
        for fut in as_completed(futures):
            results.extend(fut.result())

    results.sort(key=lambda x: x[0])  # (idx, pred)
    preds = [p for _, p in results]
    return preds

# ======================= Task pipelines (with target counts) =======================

def run_extended_adv_seq(vlm_model_id, devices, extended_df, adv_df,
                         concept, sd_batch, eval_batch, tmp_image_dir,
                         final_image_dir, avail_dir, seed, target_count):
    """
    extended_adv 모드: extended × adversarial 조합에서 concept-positive인 프롬프트를
    target_count 개까지 확보 (이미 존재하는 avail_prompts를 이어서 사용).
    """
    concept_key = concept  # "elon_musk"
    concept_sp = concept_key.replace('_', ' ')

    # 0) 기존 avail 불러오기
    existing_cols = ["expr", "concept", "extended_type", "adv_type"]
    existing_df, existing_exprs, csv_path = load_existing_avail(
        avail_dir, "extended_adv.csv", existing_cols
    )
    current_count = len(existing_df)
    if target_count is not None and current_count >= target_count:
        print(f"[extended_adv] concept={concept_key}: already have {current_count} >= target {target_count}. Skip.")
        return

    needed = None if target_count is None else max(0, target_count - current_count)
    print(f"[extended_adv] concept={concept_key}: current={current_count}, need={needed} more.")

    # 1) 조합 프롬프트 생성 (기존 expr 제외)
    prompts, ext_types, adv_types = [], [], []
    for _, ext_row in extended_df.iterrows():
        base_expr = str(ext_row["expr"])
        et = ext_row["type"]
        if concept_sp not in base_expr:
            continue
        for _, adv_row in adv_df.iterrows():
            adv_expr = str(adv_row["expr"])
            at = adv_row["type"]
            new_expr = base_expr.replace(concept_sp, adv_expr)
            if new_expr in existing_exprs:
                continue
            prompts.append(new_expr)
            ext_types.append(et)
            adv_types.append(at)

    if not prompts:
        print(f"[extended_adv] concept={concept_key}: no new candidate prompts left.")
        if target_count is not None and current_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "extended_adv",
                "have": current_count,
                "target": target_count,
                "reason": "no_candidates"
            })
        return

    # 2) 생성 (SD만 로드)
    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, prompts, sd_batch, tmp_image_dir, seed)
    assert len(gen_pairs) == len(prompts), "Mismatch between prompts and generated images."

    # 3) 평가 (VLM만 로드)
    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept_key, eval_batch)

    # 4) 필터링 + target_count까지 채우기
    new_rows, selected_paths = [], []
    for (img_path, expr), et, at, pred in zip(gen_pairs, ext_types, adv_types, preds):
        if pred == 1 and expr not in existing_exprs:
            existing_exprs.add(expr)
            new_rows.append((expr, concept_sp, et, at))
            selected_paths.append(img_path)
            if needed is not None and len(new_rows) >= needed:
                break

    if not new_rows:
        print(f"[extended_adv] concept={concept_key}: no new positives found.")
        final_count = current_count
        if target_count is not None and final_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "extended_adv",
                "have": final_count,
                "target": target_count,
                "reason": "no_positives"
            })
        return

    new_df = pd.DataFrame(new_rows, columns=existing_cols)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    if target_count is not None:
        updated_df = updated_df.iloc[:target_count]
    updated_df = updated_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    updated_df.to_csv(csv_path, index=False)
    final_count = len(updated_df)
    print(f"[extended_adv] concept={concept_key}: total avail={len(updated_df)}")

    if target_count is not None and final_count < target_count:
        SHORTAGE_LOG.append({
            "concept": concept_key,
            "mode": "extended_adv",
            "have": final_count,
            "target": target_count,
            "reason": "exhausted_but_under_target"
        })

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

def run_extended_seq(vlm_model_id, devices, prompts_df, concept,
                     sd_batch, eval_batch, tmp_image_dir, final_image_dir,
                     avail_dir, seed, target_count):
    """
    extended 모드: extended.csv의 expr들 중 VLM이 concept-positive라 판단한 것만
    target_count 개까지 avail_prompts/extended.csv에 유지.
    """
    concept_key = concept
    concept_sp = concept_key.replace('_', ' ')

    existing_cols = ["expr", "concept", "type"]
    existing_df, existing_exprs, csv_path = load_existing_avail(
        avail_dir, "extended.csv", existing_cols
    )
    current_count = len(existing_df)
    if target_count is not None and current_count >= target_count:
        print(f"[extended] concept={concept_key}: already have {current_count} >= target {target_count}. Skip.")
        return

    needed = None if target_count is None else max(0, target_count - current_count)
    print(f"[extended] concept={concept_key}: current={current_count}, need={needed} more.")

    # 후보 중 기존 expr는 제외
    candidates = []
    for _, row in prompts_df.iterrows():
        expr = str(row["expr"])
        typ = row["type"]
        if expr in existing_exprs:
            continue
        candidates.append((expr, typ))

    if not candidates:
        print(f"[extended] concept={concept_key}: no new candidate prompts left.")
        if target_count is not None and current_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "extended",
                "have": current_count,
                "target": target_count,
                "reason": "no_candidates"
            })
        return

    cand_prompts = [e for e, _ in candidates]
    cand_types = [t for _, t in candidates]

    # 생성
    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, cand_prompts, sd_batch, tmp_image_dir, seed)

    # 평가
    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept_key, eval_batch)

    # 필터링
    new_rows, selected_paths = [], []
    for (img_path, expr), typ, pred in zip(gen_pairs, cand_types, preds):
        if pred == 1 and expr not in existing_exprs:
            existing_exprs.add(expr)
            new_rows.append((expr, concept_sp, typ))
            selected_paths.append(img_path)
            if needed is not None and len(new_rows) >= needed:
                break

    if not new_rows:
        print(f"[extended] concept={concept_key}: no new positives found.")
        final_count = current_count
        if target_count is not None and final_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "extended",
                "have": final_count,
                "target": target_count,
                "reason": "no_positives"
            })
        return

    new_df = pd.DataFrame(new_rows, columns=existing_cols)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    if target_count is not None:
        updated_df = updated_df.iloc[:target_count]
    updated_df = updated_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    updated_df.to_csv(csv_path, index=False)
    final_count = len(updated_df)
    print(f"[extended] concept={concept_key}: total avail={len(updated_df)}")

    if target_count is not None and final_count < target_count:
        SHORTAGE_LOG.append({
            "concept": concept_key,
            "mode": "extended",
            "have": final_count,
            "target": target_count,
            "reason": "exhausted_but_under_target"
        })

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

def run_short_adv_seq(vlm_model_id, devices, prompts_df, task, concept,
                      sd_batch, eval_batch, tmp_image_dir, final_image_dir,
                      avail_dir, seed, target_count):
    """
    short_adv 모드: adversarial.csv의 expr에 BASE_FORMATS를 입힌 프롬프트들 중
    concept-positive인 것만 target_count 개까지 확보.
    """
    concept_key = concept
    concept_sp = concept_key.replace('_', ' ')
    base_formats = BASE_FORMATS.get(task, [])
    if not base_formats:
        print(f"[short_adv] task={task}: BASE_FORMATS가 비어 있습니다. Skip.")
        return

    existing_cols = ["expr", "concept", "type"]
    existing_df, existing_exprs, csv_path = load_existing_avail(
        avail_dir, "short_adv.csv", existing_cols
    )
    current_count = len(existing_df)
    if target_count is not None and current_count >= target_count:
        print(f"[short_adv] concept={concept_key}: already have {current_count} >= target {target_count}. Skip.")
        return

    needed = None if target_count is None else max(0, target_count - current_count)
    print(f"[short_adv] concept={concept_key}: current={current_count}, need={needed} more.")

    # 후보 생성 (기존 expr 제외)
    all_prompts, all_types = [], []
    for _, row in prompts_df.iterrows():
        expr = str(row["expr"])
        typ = row["type"]
        for base in base_formats:
            formatted_expr = base + expr
            if formatted_expr in existing_exprs:
                continue
            all_prompts.append(formatted_expr)
            all_types.append(typ)
    all_prompts = all_prompts[:64]
    all_types = all_types[:64]

    if not all_prompts:
        print(f"[short_adv] concept={concept_key}: no new candidate prompts left.")
        if target_count is not None and current_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "short_adv",
                "have": current_count,
                "target": target_count,
                "reason": "no_candidates"
            })
        return

    # 생성
    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, all_prompts, sd_batch, tmp_image_dir, seed)

    # 평가
    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept_key, eval_batch)

    # 필터링
    new_rows, selected_paths = [], []
    for (img_path, formatted_expr), typ, pred in zip(gen_pairs, all_types, preds):
        if pred == 1 and formatted_expr not in existing_exprs:
            existing_exprs.add(formatted_expr)
            new_rows.append((formatted_expr, concept_sp, typ))
            selected_paths.append(img_path)
            if needed is not None and len(new_rows) >= needed:
                break

    if not new_rows:
        print(f"[short_adv] concept={concept_key}: no new positives found.")
        final_count = current_count
        if target_count is not None and final_count < target_count:
            SHORTAGE_LOG.append({
                "concept": concept_key,
                "mode": "short_adv",
                "have": final_count,
                "target": target_count,
                "reason": "no_positives"
            })
        return

    new_df = pd.DataFrame(new_rows, columns=existing_cols)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    if target_count is not None:
        updated_df = updated_df.iloc[:target_count]
    updated_df = updated_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    updated_df.to_csv(csv_path, index=False)
    final_count = len(updated_df)
    print(f"[short_adv] concept={concept_key}: total avail={len(updated_df)}")

    if target_count is not None and final_count < target_count:
        SHORTAGE_LOG.append({
            "concept": concept_key,
            "mode": "short_adv",
            "have": final_count,
            "target": target_count,
            "reason": "exhausted_but_under_target"
        })

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

# ======================= Main =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=str, default="OpenGVLab/InternVL3-38B-hf")
    parser.add_argument("--output_path", type=str, default="./data/avail_prompts")
    parser.add_argument("--prompt_path", type=str, default="./data/prompt_pool")
    parser.add_argument("--task", type=str, default="celebrity")
    parser.add_argument("--concepts", type=str, required=True,
                    help="여러 concept을 쉼표로 구분하여 입력 (예: \"elon musk,bill gates\")")
    parser.add_argument("--sd_batch", type=int, default=64)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target_short_adv", type=int, default=55,
                        help="short_adv에서 확보할 avail_prompts 개수 (0이면 실행 안 함)")
    parser.add_argument("--target_extended", type=int, default=95,
                        help="extended에서 확보할 avail_prompts 개수 (0이면 실행 안 함)")
    parser.add_argument("--target_extended_adv", type=int, default=120,
                        help="extended_adv에서 확보할 avail_prompts 개수 (0이면 실행 안 함)")
    args = parser.parse_args()

    devices = pick_devices()

    # concept 문자열 정규화
    raw = args.concepts
    concepts = [c.strip() for c in raw.split(",") if c.strip()]
    concepts = [c.replace(" ", "_").lower() for c in concepts]

    for concept in concepts:
        print(f"\n======================= Working on concept [{concept}] =======================")

        # 경로
        output_dir = os.path.join(args.output_path, args.task, concept)
        prompt_dir = os.path.join(args.prompt_path, args.task, concept)
        avail_prompts_path = output_dir
        avail_images_path  = os.path.join(output_dir, "avail_images")
        tmp_image_path     = os.path.join(output_dir, "tmp_image_path")

        ensure_dir(output_dir)
        ensure_dir(avail_prompts_path)
        ensure_dir(avail_images_path)
        ensure_dir(tmp_image_path)

        # 프롬프트 로드 (extended / adversarial)
        adv_csv_path = os.path.join(prompt_dir, "adversarial.csv")
        extended_csv_path = os.path.join(prompt_dir, "extended.csv")

        adv_prompts_df = None
        extended_prompts_df = None

        if os.path.exists(adv_csv_path):
            adv_prompts_df = pd.read_csv(adv_csv_path)
        else:
            print(f"[Warning] adversarial.csv not found for concept={concept} at {adv_csv_path}")

        if os.path.exists(extended_csv_path):
            extended_prompts_df = pd.read_csv(extended_csv_path)
        else:
            print(f"[Warning] extended.csv not found for concept={concept} at {extended_csv_path}")

        # short_adv
        if args.target_short_adv > 0:
            if adv_prompts_df is None:
                print(f"[short_adv] Skip concept={concept}: adversarial.csv missing.")
            else:
                run_short_adv_seq(
                    vlm_model_id=args.vlm,
                    devices=devices,
                    prompts_df=adv_prompts_df,
                    task=args.task,
                    concept=concept,
                    sd_batch=args.sd_batch,
                    eval_batch=args.eval_batch,
                    tmp_image_dir=tmp_image_path,
                    final_image_dir=avail_images_path,
                    avail_dir=avail_prompts_path,
                    seed=args.seed,
                    target_count=args.target_short_adv,
                )

        # extended
        if args.target_extended > 0:
            if extended_prompts_df is None:
                print(f"[extended] Skip concept={concept}: extended.csv missing.")
            else:
                run_extended_seq(
                    vlm_model_id=args.vlm,
                    devices=devices,
                    prompts_df=extended_prompts_df,
                    concept=concept,
                    sd_batch=args.sd_batch,
                    eval_batch=args.eval_batch,
                    tmp_image_dir=tmp_image_path,
                    final_image_dir=avail_images_path,
                    avail_dir=avail_prompts_path,
                    seed=args.seed,
                    target_count=args.target_extended,
                )

        # extended_adv
        if args.target_extended_adv > 0:
            if extended_prompts_df is None or adv_prompts_df is None:
                print(f"[extended_adv] Skip concept={concept}: needed CSVs missing.")
            else:
                run_extended_adv_seq(
                    vlm_model_id=args.vlm,
                    devices=devices,
                    extended_df=extended_prompts_df,
                    adv_df=adv_prompts_df,
                    concept=concept,
                    sd_batch=args.sd_batch,
                    eval_batch=args.eval_batch,
                    tmp_image_dir=tmp_image_path,
                    final_image_dir=avail_images_path,
                    avail_dir=avail_prompts_path,
                    seed=args.seed,
                    target_count=args.target_extended_adv,
                )

        # tmp 폴더는 남겨두고 싶으면 주석 유지, 아니면 아래 주석 해제
        # try:
        #     shutil.rmtree(tmp_image_path, ignore_errors=True)
        # except Exception as e:
        #     print(f"[Warning] Failed to remove tmp dir: {e}")

    # ===== 부족한 concept/mode 요약 출력 =====
    if SHORTAGE_LOG:
        print("\n======================= Shortage summary =======================")
        for rec in SHORTAGE_LOG:
            print(
                f"[{rec['mode']}] concept={rec['concept']}: "
                f"have {rec['have']} < target {rec['target']} "
                f"(reason={rec['reason']})"
            )
    else:
        print("\nAll concepts/modes reached their target counts (or were skipped).")
