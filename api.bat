set CUDA_VISIBLE_DEVICES=0 

python examples\api\openai_api.py --server-name 0.0.0.0 --server-port 8080 --device cuda:0 --checkpoint-path train\output 

:     train\output   D:\LLM\models\Qwen\Qwen2-7B-Instruct