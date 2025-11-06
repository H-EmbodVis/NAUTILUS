export CUDA_VISIBLE_DEVICES=1

# 1. Inference and Evaluation
python ./Eval/eval4nautilus_llava.py \
    --model-path "model_path" \
    --image-folder "nautdata_images" \
    --answer-folder "answer saving directory" \
    --metric-folder "metric saving directory" \
    --dinov2-weight "dino_vitl.pth" \
    --benchmark-json "nautilus-llava-instruct-test.json" \
    # --grounding-only

# 2. Evaluation with predictions

# python ./Eval/eval4nautilus_llava.py \
#     --image-folder "nautdata_images" \
#     --metric-folder "metric saving directory" \
#     --benchmark-json "nautilus-llava-instruct-test.json" \
#     --use-predictions \
#     --prediction-json "prediction.json"