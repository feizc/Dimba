import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.model.nets import Dimba_L_2, Dimba_G_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--model_version', default='dimba-l', type=str) # dimba-g
    parser.add_argument("--t5_path", default='ckpts/t5-v1_1-xxl', type=str,)
    parser.add_argument('--txt_file', default='asset/examples.txt', type=str)
    parser.add_argument('--model_path', default='ckpts/dimba-l-512.pt', type=str)
    parser.add_argument('--vae_path', default='ckpts/vae', type=str) 
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--step', default=20, type=int)
    parser.add_argument('--save_path', default='vis', type=str)
    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        if bs == 1:
            #save_path = os.path.join(save_root, f"{prompts[0][:100]}.jpg")
            #if os.path.exists(save_path):
            #    continue
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
                                  return_tensors="pt").to(device)
        caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        emb_masks = caption_token.attention_mask

        caption_embs = caption_embs[:, None]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        print(f'finish embedding')

        with torch.no_grad():

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = 300
    pe_interpolation = {512: 1, 1024: 2, 2048: 4}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    micro_condition = False
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting 
    if args.model_version == 'dimba-l': 
        model = Dimba_L_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = Dimba_G_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)

    print("Generating sample from ckpt: %s" % args.model_path)

    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage) 

    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).to(weight_dtype)
    t5_path = args.t5_path
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path).to(device)

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    work_dir = args.save_path

    # data setting
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # img save setting
    

    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale)
