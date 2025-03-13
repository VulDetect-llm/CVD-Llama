export OUTPUT_DIR=output
export GTE_MODEL_PATH=Alibaba-NLP/gte-large-en-v1.5
export MODEL_NAME_OR_PATH=mllama


deepspeed model/train_cvd_llama.py \
    --deepspeed ds_config.json \
    -- bf16 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --save_total_limit 2 \
    --save_strategy epoch \
    --gte_model_max_length 1024 \
    --llama_model_max_length 1024 \
    --gte_model_path $GTE_MODEL_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \

