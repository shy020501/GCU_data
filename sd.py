import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import os

NUM_INFERENCE_STEP = 50
GUIDANCE_SCALE = 5.0
ETA = 1.0

class SD:
    def __init__(self, device, model_id="CompVis/stable-diffusion-v1-4"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to(device)
        self.pipeline.safety_checker = None
        
    def create_image(self, prompts, save_dir="./generated_images", seed=42):
        os.makedirs(save_dir, exist_ok=True)

        tokenizer_output = self.pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.tokenizer.model_max_length,
        )
        input_ids = tokenizer_output.input_ids.to(self.pipeline.device)

        with torch.no_grad():
            prompt_embeds = self.pipeline.text_encoder(input_ids)[0]  # shape: (B, L, D)

        generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)

        with torch.autocast(device_type=self.pipeline.device.type):
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=NUM_INFERENCE_STEP,
                guidance_scale=GUIDANCE_SCALE,
                eta=ETA,
                output_type="pil",
                generator=generator,
            )

        save_paths = []
        
        for prompt, image in zip(prompts, result.images):
            prompt_text = prompt.replace(' ', '_')
            file_name = f"{seed}_{prompt_text}.png"
            save_path = os.path.join(save_dir, file_name)
            image.save(save_path)
            save_paths.append(save_path)
        
        return save_paths
