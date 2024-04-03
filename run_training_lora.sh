export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="Dataset"
export model_output="text-to-music-sd"

accelerate launch train_text_to_image_lora.py 
      --pretrained_model_name_or_path=$MODEL_NAME 
      --dataset_name=$dataset_name 
      --train_batch_size=50 
      --gradient_accumulation_steps=4 
      --gradient_checkpointing 
      --max_train_steps=10000 
      --learning_rate=1e-05 
      --max_grad_norm=1 
      --lr_scheduler="constant" 
      --lr_warmup_steps=0 
      --report_to="tensorboard" 
      # Mixed precision not supported on mac
      --mixed_precision="fp16" 
      --snr_gamma=5 
      --checkpointing_steps=20 
      --checkpoints_total_limit=10 
      --resume_from_checkpoint="latest" 
      --dataloader_num_workers=4 
      # Model name locally and on repo
      --output_dir=$model_output 
      --push_to_hub 
      --hub_model_id="BrijeshGiri/"+$model_output 
      --hub_token="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" 
      # Attention config
      --enable_xformers_memory_efficient_attention 
      --rank=64 
      # Speedup for Ampere GPU:A100
      --allow_tf32 
