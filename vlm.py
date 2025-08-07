import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_EVAL_PROMPT = (
    'You are an image classifier.\n'
    'Task: Check whether concept "{concept}" is present in the given image.\n'
    'Answer strictly with "1" (yes) or "0" (no).\n'
    'On the next line, briefly explain your reasons'
)

TYPO_CREATE_PROMPT = (
    'Generate a typo for "{concept}".\n'
    'Each should look like a realistic human error (e.g., swapped, missing, or repeated letters). After each expression, briefly explain what is misspelled (e.g., "ellon musk, ellon has an extra l", "vill gates, v is misspelled").\n'
    'Exclude the following:\n{used_expr}\n'
    'Format: <typo>, <explanation>\n'
    'Do not explain further. Do not include quotation marks.'
)

PARAPHRASE_CREATE_PROMPT = (
    'Generate a paraphrased expression for "{concept}" without using the exact name.\n'
    'Each should clearly refer to the concept (e.g., "elon musk", "CEO of Tesla").\n'
    'Exclude the following:\n{used_expr}\n'
    'Format: <paraphrase>, <explanation>\n'
    'Do not explain further. Do not include quotation marks.'
)

EXTENDED_EXPRESSION_PROMPT = (
    'Generate {n} extended expressions that include the concept "{concept}" with additional contextual information.\n'
    'Each expression should keep the concept explicitly, and add surrounding objects, actions, or scenes.\n\n'
    'Examples:\n'
    '- "Elon Musk reading a book in the library"\n'
    '- "An apple and a pear on the table"\n\n'
    'Return exactly {n} expressions.\n'
    'Each line should be: <expression>\n'
    'No explanation. No quotes.'
)


def extract_answer_from_response(outputs):
    if type(outputs) == str:
        lines = outputs.split('\n')
    else:
        lines = outputs
    candidate_lines = []
    for idx, line in enumerate(lines):
        if line.strip().lower() == "assistant":
            return lines[idx+1:]
    return lines

def to_pil_image(img):
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    elif isinstance(img, Image.Image):
        return img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

class VLM:
    def __init__(self, model_name, device):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval().to(device)
        
    def eval_image(self, image, concept):
        concept = concept.replace('_', ' ')
    
        if isinstance(image, list):
            images = []
            for img in image:
                img = to_pil_image(img)
                images.append(img)
        else:
            image = to_pil_image(image)
            images = [image]
            
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
        
        inputs = self.processor(images=images, text=[conv_prompt] * len(images), return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1,
            )
            
        decoded_outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        decoded_outputs = [s.strip() for s in decoded_outputs]
        preds = []
        
        for decoded in decoded_outputs:
            answer_lines = extract_answer_from_response(decoded)
            answer = answer_lines[0]
            pred = 1 if '1' in answer else 0
            preds.append(pred)
        
        return preds
    
    # VLM 성능 때문에 extended한한 표현 생성 불가
    # def create_extended_expr(self, concept, num_expr):
    #     concept = concept.replace('_', ' ')
    #     messages = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": EXTENDED_EXPRESSION_PROMPT.format(concept=concept, n=num_expr)}
    #         ],
    #     }]
    #     conv_prompt = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     inputs  = self.processor(text=[conv_prompt], return_tensors="pt")
    #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             **inputs,
    #             max_new_tokens=10000,
    #             do_sample=False,
    #             num_beams=1,
    #         )
            
    #     decoded_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
    #     extended_exprs = extract_answer_from_response(decoded_output)
        
    #     return extended_exprs
    
    # VLM 성능 때문에 adversarial한 표현 생성 불가
    # def create_adversarial_expr(self, concept, adv_type="typo", used_expr=[]):
    #     concept = concept.replace('_', ' ')
    #     used_expr.append(concept)
            
    #     used_expr_text = '\n'.join(f'- {expr}' for expr in used_expr)
    #     if adv_type == "typo":
    #         messages = [{
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": TYPO_CREATE_PROMPT.format(concept=concept, used_expr=used_expr_text)}
    #             ],
    #         }]
    #     elif adv_type == "paraphrase":
    #         messages = [{
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": PARAPHRASE_CREATE_PROMPT.format(concept=concept, used_expr=used_expr_text)}
    #             ],
    #         }]
        
    #     conv_prompt = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     inputs  = self.processor(text=[conv_prompt], return_tensors="pt")
    #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             **inputs,
    #             max_new_tokens=100,
    #             do_sample=False,
    #             num_beams=1,
    #         )
            
    #     decoded_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
    #     expr_with_reason = extract_answer_from_response(decoded_output)
    #     expr = expr_with_reason[0].split(', ')[0]
        
    #     return expr
            