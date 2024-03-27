export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="sadrasabouri/ShahNegar"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=4 \
#  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
#  --enable_xformers_memory_efficient_attention
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="farsi-poetry-model" \
  --push_to_hub
#  --mixed_precision="fp16"