export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="sadrasabouri/ShahNegar"

accelerate launch train_text_to_image_lora.py \
      --train_batch_size=1
      --gradient_accumulation_steps=4
      --gradient_checkpointing
      --max_train_steps=15
      --learning_rate=1e-05
      --max_grad_norm=1
      --lr_scheduler="constant"
      --lr_warmup_steps=0
      --output_dir="base-music-model"
      --report_to="wandb"
      --mixed_precision="fp16"
      --snr_gamma=5
      --checkpointing_steps=5
      --checkpoints_total_limit=5
      --resume_from_checkpoint="latest"
      --dataloader_num_workers=4