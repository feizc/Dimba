data_path = '/path/to/data'
image_size = 512

# model setting
model = 'Dimba_L_2'
mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
load_from = "ckpts/dimba-l-512.pt" 
vae_pretrained = "ckpts/vae"  
t5_path = "ckpts/t5-v1_1-xxl"
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 0
train_batch_size = 16  # 48 as default
num_epochs = 5  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
use_fsdp=False   # if use FSDP mode
kv_compress = False
qk_norm = False 
micro_condition = False
snr_loss=False
gradient_clip = 0.05
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)
lr_schedule = 'constant'

eval_sampling_steps = 500000
visualize = False
log_interval = 20
save_model_epochs = 5
save_model_steps = 5000
work_dir = 'output/debug'

scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.1
train_sampling_steps = 1000
skip_step=0
sample_posterior = True