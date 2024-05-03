def filter_duplicate_concepts(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if line.split('||')[1] != line.split('||')[2].strip()]

    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

# Loop over the 10 files
for i in range(10):
    input_file = f'Replace with path that stores the training dataset for AskAPatient or TwADR-L' 
    output_file = f'Replace with path that new dataset need to be store'  # Output file path
    filter_duplicate_concepts(input_file, output_file)
