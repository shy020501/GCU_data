# main.py (phase-wise: SD-only during generation, VLM-only during evaluation; VLM uses device_map="auto")
import os
import gc
import shutil
import argparse
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from vlm import VLM
from sd import SD

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_FORMATS = {
    'celebrity': ["Photo of ", "Image of ", "Portrait of ", "Close-up shot of ", "Realistic rendering of "],
    'style': ["A painting by ", "Art by ", "Artwork by ", "Picture by ", "Style of "]
}

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
    """선택된 이미지만 최종 폴더로 이동하고 나머지는 삭제."""
    ensure_dir(final_image_dir)
    selected_abs = set(os.path.abspath(p) for p in selected_paths)

    # 삭제
    # for fname in os.listdir(tmp_image_dir):
    #     if not fname.lower().endswith(".png"):
    #         continue
    #     fpath = os.path.abspath(os.path.join(tmp_image_dir, fname))
    #     if fpath not in selected_abs:
    #         try:
    #             os.remove(fpath)
    #         except Exception as e:
    #             print(f"[Warning] Failed to delete {fname}: {e}")

    # 이동
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
        # VLM 래퍼가 **kwargs를 내부 HF 로더로 전달해야 합니다.
        self.pipes = [VLM(self.model_id, device=None, device_map="auto")]
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
    """단일 디바이스에서 [(idx, path)] 평가 → [(idx, pred)] 리턴 (배치 단위 backoff 포함)
    device_map='auto'에서는 모델이 내부적으로 여러 GPU를 사용하므로 set_device를 건드리지 않습니다.
    """
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

# ======================= Task pipelines (phase-wise load/unload) =======================

def run_extended_adv_seq(vlm_model_id, devices, extended_df, adv_df,
                         concept, sd_batch, eval_batch, tmp_image_dir,
                         final_image_dir, avail_dir, seed):
    concept_sp = concept.replace('_', ' ')
    prompts, ext_types, adv_types = [], [], []

    for _, ext_row in extended_df.iterrows():
        base_expr = ext_row["expr"]; et = ext_row["type"]
        for _, adv_row in adv_df.iterrows():
            adv_expr = adv_row["expr"]; at = adv_row["type"]
            if concept_sp not in base_expr:
                continue
            prompts.append(base_expr.replace(concept_sp, adv_expr))
            ext_types.append(et)
            adv_types.append(at)

    # 1) 생성: SD만 로드
    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, prompts, sd_batch, tmp_image_dir, seed)
    assert len(gen_pairs) == len(prompts), "Mismatch between prompts and generated images."

    # 2) 평가: VLM만 로드 (device_map='auto', 단일 인스턴스)
    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept, eval_batch)

    # 3) 필터/저장/정리
    filtered_rows, selected_paths = [], []
    for (img_path, expr), et, at, pred in zip(gen_pairs, ext_types, adv_types, preds):
        if pred == 1:
            filtered_rows.append((expr, concept_sp, et, at))
            selected_paths.append(img_path)

    ensure_dir(avail_dir)
    if filtered_rows:
        pd.DataFrame(filtered_rows, columns=["expr", "concept", "extended_type", "adv_type"])\
          .to_csv(os.path.join(avail_dir, "extended_adv.csv"), index=False)

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

def run_extended_seq(vlm_model_id, devices, prompts_df, concept,
                     sd_batch, eval_batch, tmp_image_dir, final_image_dir, avail_dir, seed):
    prompts = prompts_df["expr"].tolist()
    types   = prompts_df["type"].tolist()

    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, prompts, sd_batch, tmp_image_dir, seed)

    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept, eval_batch)

    concept_sp = concept.replace('_', ' ')
    filtered_rows, selected_paths = [], []
    for (img_path, expr), typ, pred in zip(gen_pairs, types, preds):
        if pred == 1:
            filtered_rows.append((expr, concept_sp, typ))
            selected_paths.append(img_path)

    ensure_dir(avail_dir)
    if filtered_rows:
        pd.DataFrame(filtered_rows, columns=["expr", "concept", "type"])\
          .to_csv(os.path.join(avail_dir, "extended.csv"), index=False)

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

def run_short_adv_seq(vlm_model_id, devices, prompts_df, task,
                      sd_batch, eval_batch, tmp_image_dir, final_image_dir, avail_dir, seed):
    concept = prompts_df["concept"].iloc[0]
    base_formats = BASE_FORMATS.get(task, [])
    all_prompts, all_types = [], []

    for _, row in prompts_df.iterrows():
        expr = row["expr"]; typ = row["type"]
        for base in base_formats:
            all_prompts.append(base + expr)
            all_types.append(typ)

    with SDGroup(devices) as sd_pipes:
        gen_pairs = parallel_generate_all(sd_pipes, all_prompts, sd_batch, tmp_image_dir, seed)

    with VLMGroup(vlm_model_id, devices) as vlm_pipes:
        preds = parallel_eval_all(vlm_pipes, gen_pairs, concept, eval_batch)

    concept_sp = concept.replace('_', ' ')
    filtered_rows, selected_paths = [], []
    for (img_path, formatted_expr), typ, pred in zip(gen_pairs, all_types, preds):
        if pred == 1:
            filtered_rows.append((formatted_expr, concept_sp, typ))
            selected_paths.append(img_path)

    ensure_dir(avail_dir)
    if filtered_rows:
        pd.DataFrame(filtered_rows, columns=["expr", "concept", "type"])\
          .to_csv(os.path.join(avail_dir, "short_adv.csv"), index=False)

    move_selected_and_cleanup(selected_paths, tmp_image_dir, final_image_dir)

# ======================= Main =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=str, default="OpenGVLab/InternVL3-38B-hf")
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--prompt_path", type=str, default="./data/prompts")
    parser.add_argument("--task", type=str, default="celebrity")
    parser.add_argument("--concept", type=str, default="elon musk")
    parser.add_argument("--sd_batch", type=int, default=64)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="extended_adv",
                        choices=["extended_adv", "extended", "short_adv"],
                        help="실행할 파이프라인 선택")
    args = parser.parse_args()

    args.concept = args.concept.replace(' ', '_').lower()
    devices = pick_devices()

    # 경로
    output_dir = os.path.join(args.output_path, args.task, args.concept)
    prompt_dir = os.path.join(args.prompt_path, args.task, args.concept)
    avail_prompts_path = os.path.join(output_dir, "avail_prompts")
    avail_images_path  = os.path.join(output_dir, "avail_images")
    tmp_image_path     = os.path.join(output_dir, "tmp_image_path")

    ensure_dir(output_dir)
    ensure_dir(avail_prompts_path)
    ensure_dir(avail_images_path)
    ensure_dir(tmp_image_path)

    # 프롬프트 로드
    adv_csv_path = os.path.join(prompt_dir, "adversarial.csv")
    extended_csv_path = os.path.join(prompt_dir, "extended.csv")
    adv_prompts_df = pd.read_csv(adv_csv_path)
    extended_prompts_df = pd.read_csv(extended_csv_path)

    # ====== Phase-wise execution (SD only -> VLM only w/ device_map='auto') ======
    
    print(f"======================= Working on [{args.mode}] =======================")

    if args.mode == "short_adv":
        run_short_adv_seq(
            vlm_model_id=args.vlm,
            devices=devices,
            prompts_df=adv_prompts_df,
            task=args.task,
            sd_batch=args.sd_batch,
            eval_batch=args.eval_batch,
            tmp_image_dir=tmp_image_path,
            final_image_dir=avail_images_path,
            avail_dir=avail_prompts_path,
            seed=args.seed,
        )

    elif args.mode == "extended":
        run_extended_seq(
            vlm_model_id=args.vlm,
            devices=devices,
            prompts_df=extended_prompts_df,
            concept=args.concept,
            sd_batch=args.sd_batch,
            eval_batch=args.eval_batch,
            tmp_image_dir=tmp_image_path,
            final_image_dir=avail_images_path,
            avail_dir=avail_prompts_path,
            seed=args.seed,
        )

    elif args.mode == "extended_adv":
        run_extended_adv_seq(
            vlm_model_id=args.vlm,
            devices=devices,
            extended_df=extended_prompts_df,
            adv_df=adv_prompts_df,
            concept=args.concept,
            sd_batch=args.sd_batch,
            eval_batch=args.eval_batch,
            tmp_image_dir=tmp_image_path,
            final_image_dir=avail_images_path,
            avail_dir=avail_prompts_path,
            seed=args.seed,
        )

    # 임시 폴더 정리
    # try:
    #     shutil.rmtree(tmp_image_path, ignore_errors=True)
    # except Exception as e:
    #     print(f"[Warning] Failed to remove tmp dir: {e}")
