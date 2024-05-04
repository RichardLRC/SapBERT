DICT_PATH="Path to AskAPatient.dict.txt or TwADR-L.dict.txt"
sample_sizes=(1 5 10 20 40 80)

for sample_size in "${sample_sizes[@]}"
do
    for bootstrap_index in {0..19}
    do
        for VAR in 0 1 2 3 4 5 6 7 8 9
        do 
            DATA_DIR="Path to Test Dataset (AskAPatient or TwADR-L)"
	        MODEL_DIR="Path to the model directory"
            OUTPUT_DIR="Desire output directory"

            # Create the output directory if it doesn't exist
            mkdir -p $OUTPUT_DIR

            # Modify the eval.py to the correct path
            CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
                --model_dir $MODEL_DIR \
                --dictionary_path $DICT_PATH \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR/ \
                --use_cuda \
                --max_length 25 \
                --save_predictions \
                --custom_query_loader \
                --fold_number $VAR
        done
    done
done
