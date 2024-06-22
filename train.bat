
set MODEL="D:\LLM\models\Qwen\Qwen2-7B-Instruct" 
set DATA="trian\train.json"
set OUTPUT="trian\output"
set DS_CONFIG_PATH="ds_config_zero2.json"
set USE_LORA=True
set Q_LORA=True

set DS_BUILD_AIO=0
set DS_BUILD_EVOFORMER_ATTN=0
set DS_BUILD_SPARSE_ATTN=0
set DS_BUILD_OPS=0


python examples\sft\finetune.py ^
    --model_name_or_path %MODEL% ^
    --data_path %DATA% ^
    --bf16 True ^
    --output_dir %OUTPUT% ^
    --num_train_epochs 5 ^
    --per_device_train_batch_size 2 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --evaluation_strategy "no" ^
    --save_strategy "steps" ^
    --save_steps 10 ^
    --save_total_limit 10 ^
    --learning_rate 3e-4 ^
    --weight_decay 0.01 ^
    --adam_beta2 0.95 ^
    --warmup_ratio 0.01 ^
    --lr_scheduler_type "cosine" ^
    --logging_steps 1 ^
    --report_to "none" ^
    --model_max_length 512 ^
    --lazy_preprocess True ^
    --use_lora %USE_LORA% ^
    --q_lora %Q_LORA% ^
    --gradient_checkpointing ^
    --deepspeed %DS_CONFIG_PATH%
