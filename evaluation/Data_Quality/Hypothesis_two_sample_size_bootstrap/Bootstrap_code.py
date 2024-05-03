import os
import random
from collections import defaultdict

def read_and_group_data(file_path):
    data_by_concept = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split("||")
            if len(parts) == 3:
                concept_id = parts[0]
                data_by_concept[concept_id].append(line.strip())
    return data_by_concept

def create_bootstrap_files(data_by_concept, sample_sizes, n_bootstrap, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for size in sample_sizes:
        size_dir = os.path.join(output_dir, f'sample_size_{size}')
        if not os.path.exists(size_dir):
            os.makedirs(size_dir)

        for i in range(n_bootstrap):
            file_path = os.path.join(size_dir, f'bootstrap_{i}.txt')
            with open(file_path, 'w') as file:
                for concept_id, phrases in data_by_concept.items():
                    sample = random.choices(phrases, k=size)
                    for line in sample:
                        file.write(line + '\n')

# File path for your data
# Should be the path to the datasets file that contain entire generated data.
# For example, for AskAPatient, this would be the path to the 'AskAPatient.txt' file. 
# The AskAPatient dataset originally contains 1036 unique medical concepts. Then the AskAPatient.txt file should contain 103,600 lines, if 100 informal phrases are generated.
# The AskAPatient.txt file should be in the format: concept_id||phrase||label
data_file_path = 'Replace with path that stores the training dataset for AskAPatient or TwADR-L'
data_by_concept = read_and_group_data(data_file_path)

# Sample sizes and number of bootstrap iterations
sample_sizes = [1, 5, 10, 20, 40, 80]
n_bootstrap = 100
output_dir = 'Replace with path to the desired directory'  # Update this path to your desired directory

# Create bootstrap sample files in their respective folders
create_bootstrap_files(data_by_concept, sample_sizes, n_bootstrap, output_dir)

print("Bootstrap sample files created for each sample size, stored in separate folders.")
