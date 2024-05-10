# A script that evaluates all checkpoints in the checkpoint_dir, all outputs are saved in the corresponding checkpoint_dir
checkpoint_dir="dbgpt_hub/output/adapter/Meta-Llama-3-8B-Instruct-lora-20e" # CodeLlama-7b-sql-lora-11e"

current_date=$(date +"%Y%m%d_%H%M")
pred_log="dbgpt_hub/output/logs/pred_${current_date}.log"

# Iterate through each folder in the specified directory
for folder in "$checkpoint_dir"/*; do
    # Check if the folder name matches the pattern "checkpoint-XXXX"
    if [[ -d "$folder" && "$(basename "$folder")" =~ ^checkpoint-[0-9]+$ ]]; then
        echo "Running Python script for folder: $folder"
        # Run your Python script for this folder
        # python your_python_script.py "$folder"

        # Prediction
        start_time=$(date +%s)
        echo " Pred Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${pred_log}

        CUDA_VISIBLE_DEVICES=0,1  python dbgpt_hub/predict/predict.py \
            --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
            --template llama2 \
            --finetuning_type lora \
            --predicted_input_filename dbgpt_hub/data/example_text2sql_dev.json \
            --checkpoint_dir $folder \
            --predicted_out_filename "${folder}/pred.sql" >> ${pred_log}

        echo "############pred end###############" >>${pred_log}
        echo "pred End time: $(date)" >>${pred_log}
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        min=$(( (duration % 3600) / 60))
        echo "Time elapsed: ${hour}  hour $min min " >>${pred_log}

        # Evaluation
        python dbgpt_hub/eval/evaluation.py \
            --input "${folder}/pred.sql" \
            --gold "dbgpt_hub/data/eval_data/gold.txt" \
            --db "dbgpt_hub/data/spider/database" \
            --table "dbgpt_hub/data/eval_data/tables.json" \
            --etype "exec" \
            --plug_value >> ${pred_log}
    fi
done