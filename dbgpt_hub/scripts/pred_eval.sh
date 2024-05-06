folder="dbgpt_hub/output/pred"
experiment_name="codellama7b_baseline"

echo "Running Python script for folder: $folder"

# Prediction
current_date=$(date +"%Y%m%d_%H%M")
pred_log="${folder}_${current_date}.log"
start_time=$(date +%s)
echo " Pred Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${pred_log}

CUDA_VISIBLE_DEVICES=0,1  python dbgpt_hub/predict/predict.py \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --template llama2 \
    --finetuning_type lora \
    --predicted_input_filename dbgpt_hub/data/example_text2sql_dev.json \
    --predicted_out_filename "${folder}/${experiment_name}_pred.sql" >> ${pred_log}
    # --checkpoint_dir "path_to_model_checkpoint_folder" \

echo "############pred end###############" >>${pred_log}
echo "pred End time: $(date)" >>${pred_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hour}  hour $min min " >>${pred_log}

# Evaluation
python dbgpt_hub/eval/evaluation.py \
    --input "${folder}/${experiment_name}_pred.sql" \
    --gold "dbgpt_hub/data/eval_data/gold.txt" \
    --db "dbgpt_hub/data/spider/database" \
    --table "dbgpt_hub/data/eval_data/tables.json" \
    --etype "exec" \
    --plug_value \
