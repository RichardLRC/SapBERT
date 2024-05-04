DICT_PATH="Path to AskAPatient.dict.txt or TwADR-L.dict.txt"

for VAR in 0 1 2 3 4 5 6 7 8 9
do 
	DATA_DIR="Path to Test Dataset (AskAPatient or TwADR-L)"
	MODEL_DIR="Path to the model directory"

	# Modify the eval.py to the correct path
	CUDA_VISIBLE_DEVICES=$1 python3 eval.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \ 
	--output_dir  "Desire output directory" \
	--use_cuda \
	--max_length 25 \
	--save_predictions \
	--custom_query_loader \
	--fold_number $VAR # Add this line
done
