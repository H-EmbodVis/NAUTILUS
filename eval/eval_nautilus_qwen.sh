export CUDA_VISIBLE_DEVICES=0

# 1. Inference and Evaluation
python ./Eval/eval4nautilus_qwen.py \
    --checkpoint "model_path" \
    --data-path "nautdata_images" \
    --answer-folder "answer saving directory" \
    --metric-folder "metric saving directory" \
    --benchmark-json "nautilus-qwen-instruct-test.json"
    # --grounding-only

# 2. Evaluation with predictions

# python ./Eval/eval4nautilus_qwen.py \
#     --data-path "nautdata_images" \
#     --metric-folder "metric saving directory" \
#     --benchmark-json ""nautilus-qwen-instruct-test.json" \
#     --prediction-json "prediction.json"