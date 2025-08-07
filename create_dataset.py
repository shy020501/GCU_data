import os
import shutil
import pandas as pd 
import argparse
import torch
import threading
import queue
from time import sleep
from vlm import VLM
from sd import SD

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

BASE_FORMATS = {
    'celebrity': ["Photo of ", "Image of ", "Portrait of ", "Close-up shot of ", "Realistic rendering of "],
    'style': ["A painting by ", "Art by ", "Artwork by ", "Picture by ", "Style of "]
}

def move_and_cleanup_images(filtered_rows, tmp_image_dir, image_dir, seed):
    kept_files = {
        f"{seed}_{row[0].replace(' ', '_')}.png"
        for row in filtered_rows
    }

    for fname in os.listdir(tmp_image_dir):
        if not fname.endswith(".png"):
            continue
        src = os.path.join(tmp_image_dir, fname)
        dst = os.path.join(image_dir, fname)
        if fname in kept_files:
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"[Warning] Failed to move {fname}: {e}")
        else:
            try:
                os.remove(src)
            except Exception as e:
                print(f"[Warning] Failed to delete {fname}: {e}")

def create_extended_adv(eval_pipeline, sd_pipeline, extended_df, adv_df, concept, sd_batch, eval_batch, tmp_image_dir, image_dir, avail_dir, seed=42):
    prompts = []
    extended_types = []
    adv_types = []
    
    concept = concept.replace('_', ' ')

    for _, ext_row in extended_df.iterrows():
        base_expr = ext_row["expr"]
        ext_type = ext_row["type"]

        for _, adv_row in adv_df.iterrows():
            adv_expr = adv_row["expr"]
            adv_type = adv_row["type"]

            if concept not in base_expr:
                continue  # just in case
            modified = base_expr.replace(concept, adv_expr)
            prompts.append(modified)
            extended_types.append(ext_type)
            adv_types.append(adv_type)
            
    print(prompts)

    q = queue.Queue(maxsize=len(prompts))
    filtered_rows = []

    # Producer
    def producer():
        for i in range(0, len(prompts), sd_batch):
            batch_prompts = prompts[i:i+sd_batch]
            batch_ext_types = extended_types[i:i+sd_batch]
            batch_adv_types = adv_types[i:i+sd_batch]
            image_paths = sd_pipeline.create_image(batch_prompts, save_dir=tmp_image_dir, seed=seed)
            for path, prompt, ext_type, adv_type in zip(image_paths, batch_prompts, batch_ext_types, batch_adv_types):
                q.put((path, prompt, ext_type, adv_type))
        q.put(None)

    # Consumer
    def consumer():
        buffer = []
        while True:
            item = q.get()
            if item is None:
                if buffer:
                    paths, exprs, ext_types, adv_types_ = zip(*buffer)
                    preds = eval_pipeline.eval_image(list(paths), concept)
                    for expr, etype, atype, pred in zip(exprs, ext_types, adv_types_, preds):
                        if pred == 1:
                            filtered_rows.append((expr, concept, etype, atype))
                break

            buffer.append(item)
            if len(buffer) >= eval_batch:
                paths, exprs, ext_types, adv_types_ = zip(*buffer)
                preds = eval_pipeline.eval_image(list(paths), concept)
                for expr, etype, atype, pred in zip(exprs, ext_types, adv_types_, preds):
                    if pred == 1:
                        filtered_rows.append((expr, concept, etype, atype))
                buffer.clear()

    # Run threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()

    # Save
    if filtered_rows:
        df = pd.DataFrame(filtered_rows, columns=["expr", "concept", "extended_type", "adv_type"])
        df.to_csv(os.path.join(avail_dir, "extended_adv.csv"), index=False)
    
    move_and_cleanup_images(filtered_rows, tmp_image_dir, image_dir, seed)

def create_extended(eval_pipeline, sd_pipeline, prompts_df, concept, sd_batch, eval_batch, tmp_image_dir, image_dir, avail_dir, seed=42):
    prompts = prompts_df["expr"].tolist()
    types = prompts_df["type"].tolist()

    q = queue.Queue(maxsize=len(prompts))
    filtered_rows = []

    # Producer
    def producer():
        for i in range(0, len(prompts), sd_batch):
            batch_prompts = prompts[i:i+sd_batch]
            batch_types = types[i:i+sd_batch]
            image_paths = sd_pipeline.create_image(batch_prompts, save_dir=tmp_image_dir, seed=seed)
            for path, prompt, typ in zip(image_paths, batch_prompts, batch_types):
                q.put((path, prompt, typ))
        q.put(None)

    # Consumer
    def consumer():
        buffer = []
        while True:
            item = q.get()
            if item is None:
                if buffer:
                    paths, exprs, types_ = zip(*buffer)
                    preds = eval_pipeline.eval_image(list(paths), concept)
                    for expr, typ, pred in zip(exprs, types_, preds):
                        if pred == 1:
                            filtered_rows.append((expr, concept, typ))
                break

            buffer.append(item)
            if len(buffer) >= eval_batch:
                paths, exprs, types_ = zip(*buffer)
                preds = eval_pipeline.eval_image(list(paths), concept)
                for expr, typ, pred in zip(exprs, types_, preds):
                    if pred == 1:
                        filtered_rows.append((expr, concept, typ))
                buffer.clear()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()

    # Save
    if filtered_rows:
        df = pd.DataFrame(filtered_rows, columns=["expr", "concept", "type"])
        df.to_csv(os.path.join(avail_dir, "extended.csv"), index=False)

    move_and_cleanup_images(filtered_rows, tmp_image_dir, image_dir, seed)

def create_short_adv(eval_pipeline, sd_pipeline, prompts_df, task, sd_batch, eval_batch, tmp_image_dir, image_dir, avail_dir, seed=42):
    concept = prompts_df["concept"].iloc[0]
    base_formats = BASE_FORMATS.get(task, [])
    
    all_prompts = []
    all_exprs = []
    all_types = []

    for _, row in prompts_df.iterrows():
        expr = row["expr"]
        typ = row["type"]
        for base in base_formats:
            all_prompts.append(base + expr)
            all_exprs.append(expr)
            all_types.append(typ)

    q = queue.Queue(maxsize=len(all_prompts))
    filtered_rows = []
    
    def producer(sd_pipeline, prompts, exprs, types, tmp_image_dir, q, batch_size):
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_types = types[i:i+batch_size]
            
            image_paths = sd_pipeline.create_image(batch_prompts, save_dir=tmp_image_dir)
            for path, f_prompt, typ in zip(image_paths, batch_prompts, batch_types):
                q.put((path, f_prompt, typ))
        
        q.put(None)


    def consumer(eval_pipeline, q, concept, filtered_rows, batch_size):
        buffer = []

        while True:
            item = q.get()
            if item is None:
                if buffer:
                    paths, formatted_exprs, types = zip(*buffer)
                    preds = eval_pipeline.eval_image(list(paths), concept)
                    for path, f_expr, typ, pred in zip(paths, formatted_exprs, types, preds):
                        if pred == 1:
                            filtered_rows.append((f_expr, concept, typ))
                break

            buffer.append(item)
            if len(buffer) >= batch_size:
                paths, formatted_exprs, types = zip(*buffer)
                preds = eval_pipeline.eval_image(list(paths), concept)
                for path, f_expr, typ, pred in zip(paths, formatted_exprs, types, preds):
                    if pred == 1:
                        filtered_rows.append((f_expr, concept, typ))
                buffer.clear()

    producer_thread = threading.Thread(target=producer, args=(sd_pipeline, all_prompts, all_exprs, all_types, tmp_image_dir, q, sd_batch))
    consumer_thread = threading.Thread(target=consumer, args=(eval_pipeline, q, concept, filtered_rows, eval_batch))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    if filtered_rows:
        df = pd.DataFrame(filtered_rows, columns=["expr", "concept", "type"])
        save_path = os.path.join(avail_dir, "short_adv.csv")
        df.to_csv(save_path, index=False)
        
    move_and_cleanup_images(filtered_rows, tmp_image_dir, image_dir, seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=str, default="OpenGVLab/InternVL3-14B-hf", help="데이터 생성에 사용할 VLM, 호환성 확인 필요")
    parser.add_argument("--output_path", type=str, default="./data", help="데이터 저장할 root path (Symbolic link가 걸려있는 것을 추천)")
    parser.add_argument("--prompt_path", type=str, default="./data/prompts", help="사전에 만들어둔 adversarial, extended expressions들이 있는 경로")
    parser.add_argument("--task", type=str, default="celebrity", help="진행할 task")
    parser.add_argument("--concept", type=str, default="elon musk", help="생성할 concept")
    parser.add_argument("--sd_batch", type=int, default=64, help="SD inference batch")
    parser.add_argument("--eval_batch", type=int, default=16, help="평가용 batch (VLM, ResNet 등)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    args = parser.parse_args()

    args.concept = args.concept.replace(' ', '_').lower()
    
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        device1 = device2 = torch.device("cpu")
    elif available_gpus == 1:
        device1 = device2 = torch.device("cuda:0")
    else:
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
        
    eval_pipeline = VLM(args.vlm, device1) # VLM 외에 classifier로 평가 시에 해당 클래스만 변경
    sd_pipeline = SD(device2)
    
    output_dir = os.path.join(args.output_path, args.task, args.concept)
    prompt_dir = os.path.join(args.prompt_path, args.task, args.concept)
    
    avail_prompts_path = os.path.join(output_dir, "avail_prompts")
    avail_images_path = os.path.join(output_dir, "avail_images")
    tmp_image_path = os.path.join(output_dir, "tmp_image_path")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(avail_prompts_path, exist_ok=True)
    os.makedirs(tmp_image_path, exist_ok=True)
    os.makedirs(avail_images_path, exist_ok=True)
    
    adv_csv_path = os.path.join(prompt_dir, "adversarial.csv")
    extended_csv_path = os.path.join(prompt_dir, "extended.csv")
    
    adv_prompts_df = pd.read_csv(adv_csv_path)
    extended_prompts_df = pd.read_csv(extended_csv_path)
    
    create_short_adv(
        eval_pipeline=eval_pipeline, 
        sd_pipeline=sd_pipeline, 
        prompts_df=adv_prompts_df, 
        task=args.task, 
        sd_batch=args.sd_batch,
        eval_batch=args.eval_batch, 
        tmp_image_dir=tmp_image_path, 
        image_dir=avail_images_path,
        avail_dir=avail_prompts_path, 
        seed=args.seed
    )
    
    create_extended(
        eval_pipeline=eval_pipeline, 
        sd_pipeline=sd_pipeline, 
        prompts_df=extended_prompts_df, 
        concept=args.concept, 
        sd_batch=args.sd_batch, 
        eval_batch=args.eval_batch, 
        tmp_image_dir=tmp_image_path, 
        image_dir=avail_images_path,
        avail_dir=avail_prompts_path, 
        seed=args.seed
    )
    
    create_extended_adv(
        eval_pipeline=eval_pipeline, 
        sd_pipeline=sd_pipeline, 
        extended_df=extended_prompts_df, 
        adv_df=adv_prompts_df, 
        concept=args.concept, 
        sd_batch=args.sd_batch, 
        eval_batch=args.eval_batch, 
        tmp_image_dir=tmp_image_path, 
        image_dir=avail_images_path,
        avail_dir=avail_prompts_path, 
        seed=args.seed
    )

    shutil.rmtree(tmp_image_path, ignore_errors=True)