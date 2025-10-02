import math
import os
import random
import time
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
from bitsandbytes.optim import Adam8bit, AdamW8bit
from torchao.optim import AdamW4bit, CPUOffloadOptimizer

from torchao.float8 import convert_to_float8_training, Float8LinearConfig



import lightning as L
import pytorch_lightning as pl

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, int4_dynamic_activation_int4_weight, Float8DynamicActivationInt4WeightConfig

# import pytorch lightning
import lightning as L
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from diffusers import FluxKontextPipeline




def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    else:
        return False
    
    return True


def LoadPipeline(dtype = torch.bfloat16):
    
    pipeline = FluxKontextPipeline.from_pretrained("/home/cropy/flux_kontext", 
                                            local_files_only=True,
                                            # quantization_config=pipeline_quant_config,
                                            torch_dtype=torch.bfloat16)

    pipeline.to("cuda")


    # quantize_(
    #     pipeline.transformer,
    #     float8_dynamic_activation_float8_weight(),
    # )

    quantize_(
        pipeline.vae,
        float8_dynamic_activation_float8_weight(),
    )
    quantize_(
        pipeline.text_encoder,
        float8_dynamic_activation_float8_weight(),
    )
    quantize_(
        pipeline.text_encoder_2,
        float8_dynamic_activation_float8_weight(),
    )
    
    pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune", fullgraph=True)
    pipeline.text_encoder = torch.compile(pipeline.text_encoder, mode="max-autotune", fullgraph=True)
    pipeline.text_encoder_2 = torch.compile(pipeline.text_encoder_2, mode="max-autotune", fullgraph=True)


    return pipeline


class DiffusionModel(L.LightningModule):

    def __init__(self, 
                image_key, 
                base_lr=1e-5, 
                train_fp8=True,
                ):
        
        super().__init__()

        target_dtype = torch.bfloat16

        self.target_dtype = target_dtype
        self.image_key = image_key

        self.pipeline = LoadPipeline(self.target_dtype)
        self.model = self.pipeline.transformer.to(self.target_dtype)
        self.base_lr = base_lr
        self.train_fp8 = train_fp8

        if self.train_fp8:
            train_fp8_config = Float8LinearConfig.from_recipe_name("rowwise")
            convert_to_float8_training(self.model, config=train_fp8_config, module_filter_fn=module_filter_fn)
        
        for param in self.model.parameters():
            param.requires_grad = True

    


    def on_train_epoch_start(self):
        self.model.train()
        print("Epoch:", self.current_epoch, "Training mode:", self.model.training)

    def on_validation_epoch_start(self):
        self.model.eval()
        print("Epoch:", self.current_epoch, "Evaluation mode:", self.model.training)

 
    def configure_optimizers(self):
        
        warmup_lr = 1e-7       # starting LR
        warmup_steps = 200    # number of steps to warm up

        ps = [
            {"params": self.model.parameters()},
        ]

        optimizer = AdamW8bit(ps, lr=warmup_lr, betas=(0.9,0.95), weight_decay=0.01, eps=1e-7)


        def warmup_fn(step):
            if step < warmup_steps:
                return 1.0 + (self.base_lr / warmup_lr - 1) * step / float(warmup_steps)
            return self.base_lr / warmup_lr

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1,
            }
        }

    def training_step(self, batch, batch_idx):
        params = self.prepare_batch(batch)
        pred = self.apply_model(**params)
        loss = pred.mean()
        return loss
    
    def validation_step(self, batch, batch_idx):
        params = self.prepare_batch(batch)
        pred = self.apply_model(**params)
        loss = pred.mean()
        return loss
    

    def first_stage_encode(self, x, return_ids=False):
            
        with torch.no_grad():

            x = x * 2 - 1
            
            # encode
            image_latents = self.pipeline._encode_vae_image(image=x, generator=None)

            # normalize
            batch_size, num_channels_latents, image_latent_height, image_latent_width = image_latents.shape
            image_latents = self.pipeline._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            
            if return_ids:
                image_ids = self.pipeline._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, image_latents.device, image_latents.dtype
                )
                

        if return_ids:
            return image_latents, image_ids

        return image_latents
    
    def first_stage_decode(self, latents, height, width):
        
        with torch.no_grad():

            # unpack
            latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
            latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
            image = self.pipeline.vae.decode(latents, return_dict=False)[0]

            image = (image + 1) / 2
            image = image.clamp(0, 1)

        return image
    
    def apply_model(self, x, t, **kwargs):
        # t is in [0, 1]

        m = self.model

        if "x_cond" in kwargs:
            hidden_states = torch.cat([x, kwargs["x_cond"]], dim=1)
        else:
            hidden_states = x

        allowed_keys = [
            "hidden_states",
            "timestep",
            "guidance",
            "pooled_projections",
            "encoder_hidden_states",
            "txt_ids",
            "img_ids",
            "joint_attention_kwargs",
            "return_dict",
        ]

        kwargs2 = {k: v for k, v in kwargs.items() if k in allowed_keys}

        v_pred = m(hidden_states=hidden_states, timestep=t, return_dict=False, **kwargs2)[0].clone()

        v_pred = v_pred[:, :x.shape[1]]

        return v_pred
    
    def prepare_batch(self, batch):
        
        with torch.no_grad():
            # self.test()
            
            params = { }
            device = batch[self.image_key].device

            # ----------- Setup image inputs -----------
            # x0 => edited image to generate
            # x_cond => source image
            x0 = batch[self.image_key].to(self.target_dtype) # B x C x H x W
            x0, latent_ids = self.first_stage_encode(x0, return_ids=True) # B x L x c
            params["x"] = x0
            params["t"] = torch.from_numpy(np.random.uniform(0, 1, size=(x0.shape[0],))).to(device=device, dtype=self.target_dtype)

            cond = batch["x_cond"].to(self.target_dtype) # B x C x H x W
            cond, cond_latent_ids = self.first_stage_encode(cond, return_ids=True) # B x L x c
            cond_latent_ids[..., 0] = 1
            params["x_cond"] = cond

            latent_ids = torch.cat([latent_ids, cond_latent_ids], dim=0)  # dim 0 is sequence dimension
            params["img_ids"] = latent_ids

            # ----------- Setup text input -----------
            instruction = batch["instruction"]

            prompt, prompt_2 = instruction, instruction
            max_sequence_length = 512

            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                lora_scale=None,
            )

            params["pooled_projections"] = pooled_prompt_embeds
            params["encoder_hidden_states"] = prompt_embeds
            params["txt_ids"] = text_ids

            guidance_scale = 3.5
            if self.pipeline.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(cond.shape[0])
            else:
                guidance = None

            params["guidance"] = guidance

            return params

    def forward(self, x, timesteps):
        return None
