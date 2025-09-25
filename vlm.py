import torch
from PIL import Image
import numpy as np
from typing import Optional, List, Union
from transformers import AutoModelForImageTextToText, AutoProcessor
from accelerate.hooks import remove_hook_from_submodules
import gc

IMAGE_EVAL_PROMPT = (
    'You are an image classifier.\n'
    'Task: Check whether concept "{concept}" is present in the given image.\n'
    'Answer strictly with "1" (yes) or "0" (no).\n'
    'On the next line, briefly explain your reasons'
)

def extract_answer_from_response(outputs: Union[str, List[str]]) -> List[str]:
    if isinstance(outputs, str):
        lines = outputs.split("\n")
    else:
        lines = outputs
    for idx, line in enumerate(lines):
        if line.strip().lower() == "assistant":
            return lines[idx + 1 :]
    return lines

def to_pil_image(img):
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 3 and img.size(0) in (1, 3):
            img = img.permute(1, 2, 0)
        arr = img.numpy()
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    elif isinstance(img, Image.Image):
        return img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

class VLM:
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        device_map: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Args:
            model_name: HF repo id or local path
            device: 단일 디바이스 강제 시 사용 (device_map이 None일 때만 유효)
            device_map: 'auto'면 Accelerate로 멀티-GPU 샤딩. None이면 단일 디바이스 로드.
            **model_kwargs: from_pretrained에 그대로 전달 (low_cpu_mem_usage, trust_remote_code 등)
        """
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        torch_dtype = torch.float32

        default_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        default_kwargs.update(model_kwargs)

        if device_map == "auto":
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                device_map="auto",
                **default_kwargs,
            )
            self._sharded = True
            self._device = None
        else:
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **default_kwargs,
            )
            self.model.to(device)
            self._sharded = False
            self._device = device

        self.model.eval()

    @torch.inference_mode()
    def eval_image(self, image, concept: str, max_new_tokens: int = 100, do_sample: bool = False, num_beams: int = 1):
        concept = concept.replace("_", " ")

        if isinstance(image, list):
            images = [to_pil_image(img) for img in image]
        else:
            images = [to_pil_image(image)]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": IMAGE_EVAL_PROMPT.format(concept=concept)},
            ],
        }]
        conv_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            images=images,
            text=[conv_prompt] * len(images),
            return_tensors="pt",
        )

        if not self._sharded and self._device is not None and self._device.type != "cpu":
            inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
        )

        decoded_outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        decoded_outputs = [s.strip() for s in decoded_outputs]

        preds = []
        for decoded in decoded_outputs:
            lines = [ln.strip() for ln in extract_answer_from_response(decoded) if ln.strip()]
            answer = lines[0] if lines else ""
            # 가장 앞 글자 우선 판정 (노이즈 방지)
            first_char = answer.strip()[:1]
            if first_char == "1":
                preds.append(1)
            elif first_char == "0":
                preds.append(0)
            else:
                # fallback: 포함 여부로 판정
                preds.append(1 if "1" in answer and "0" not in answer else 0)

        return preds
    
    def __exit__(self, exc_type, exc, tb):
        try:
            for p in self.pipes or []:
                try:
                    remove_hook_from_submodules(p.model)
                except Exception:
                    pass
                try:
                    p.model.to("cpu")
                except Exception:
                    pass
                del p
        finally:
            self.pipes = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()