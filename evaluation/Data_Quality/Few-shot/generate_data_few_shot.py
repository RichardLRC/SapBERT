import openai
import pandas as pd
import os
import time
import re

# Setup OpenAI credentials
openai.api_key = "###########################"
openai.api_base = "###########################" 
openai.api_type = '###########################'
openai.api_version = '###########################' 
deployment_name = '###########################' 
print(openai.__version__)

# Dataset that contains the entire 10-fold training data
data_path = "./AskAPatient.entire.txt"

# Initialize lists to hold column data
concept_ids = []
medical_concepts = []
informal_phrases = []

# Read and parse the dataset with specified encoding
with open(data_path, 'r', encoding='ISO-8859-1') as file:
    for line in file:
        parts = line.strip().split('||')
        if len(parts) == 3:
            concept_ids.append(parts[0])
            medical_concepts.append(parts[1])
            informal_phrases.append(parts[2])

# Create a DataFrame
df = pd.DataFrame({
    'concept_id': concept_ids,
    'medical_concept': medical_concepts,
    'informal_phrase': informal_phrases
})

# Delete all rows where 'medical_concept' is the same as 'informal_phrase', case-insensitive
df_filtered = df[~df.apply(lambda x: x['medical_concept'].lower() == x['informal_phrase'].lower(), axis=1)]

# Delete all duplicate rows from the filtered DataFrame
df_filtered_no_duplicates = df_filtered.drop_duplicates()

# Calculate the number of unique 'medical_concept' and 'concept_id' in the filtered DataFrame
unique_medical_concepts = df_filtered_no_duplicates['medical_concept'].nunique()
unique_concept_ids = df_filtered_no_duplicates['concept_id'].nunique()

# Function to send request with retry mechanism
def send_request(prompt, max_tokens=300, temperature=0.9, top_p=0.8, attempts=3, timeout=900):
    retry_delay = 30  # Start with a 1-second delay
    for attempt in range(1, attempts + 1):  # Start attempt counting from 1
        try:
            print(f"Attempt {attempt}: Sending request...")
            response = openai.Completion.create(
                engine=deployment_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            )
            return response
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded, retrying in {retry_delay} seconds (Attempt {attempt})...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay for the next attempt
        except openai.error.OpenAIError as e:
            print(f"OpenAIError on attempt {attempt}: {e}")
            break  # Break on other errors
    raise Exception("Failed to send request after multiple attempts.")


# Define the base directory where you want to save the files
output_base_dir = "Replace with path to the desired directory"  # Update this to your desired output directory
# Ensure the output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Main loop to process each medical concept and save to separate files
for concept_id, group in df_filtered_no_duplicates.groupby('concept_id'):
    medical_concept = group['medical_concept'].iloc[0].replace("/", "_")  # Replace slashes to avoid path issues
    unique_phrases = group['informal_phrase'].unique().tolist()

    # Prepare few-shot learning prompt
    examples = unique_phrases[:10] if len(unique_phrases) >= 10 else unique_phrases
    prompt = "Given medical concept: '{}', generate 20 informal phrases, here are some examples of informal phrase for this medical concept:\n".format(medical_concept)
    for example in examples:
        prompt += "- {}\n".format(example)
    prompt += "Generate new informal phrases:"

    # Initialize list to store generated phrases for this concept
    output_lines = []

    # Send request
    try:
        response = send_request(prompt)
        generated_phrases = response['choices'][0]['text'].strip().split('\n')
        for phrase in generated_phrases:
            output_line = "{}||{}||{}".format(concept_id, medical_concept, phrase)
            output_lines.append(output_line)

        # Save output to a file specific to the current medical concept
        output_path = os.path.join(output_base_dir, f"{concept_id}_{medical_concept}.txt")
        with open(output_path, 'w') as file:
            for line in output_lines:
                file.write(line + '\n')
        
        print(f"Output for {medical_concept} saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred for concept {concept_id}: {e}")

unique_concept_ids_from_file = set(df_filtered_no_duplicates['concept_id'])

# Initialize a set to hold all concept IDs from the .txt files
all_concept_ids_from_txt_files = set()

# Initialize a list to hold the concept IDs of files with not exactly 20 lines
incorrect_line_concept_ids = []

# Iterate over all files in the specified folder
for filename in os.listdir(output_base_dir):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        # Extract the concept ID from the filename
        concept_id = filename.split('_')[0]
        all_concept_ids_from_txt_files.add(concept_id)
        
        # Check the number of lines in the file
        file_path = os.path.join(output_base_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if len(lines) != 20:
                incorrect_line_concept_ids.append(concept_id)

# Find missing concept IDs by comparing sets
missing_concept_ids = unique_concept_ids_from_file - all_concept_ids_from_txt_files

# Add the missing concept IDs to the list of incorrect lines if not already present
for concept_id in missing_concept_ids:
    if concept_id not in incorrect_line_concept_ids:
        incorrect_line_concept_ids.append(concept_id)

# Print the results
print(f"Total unique concept IDs from .txt files: {len(all_concept_ids_from_txt_files)}")
print("Missing concept IDs added to incorrect_line_concept_ids list:")
print(missing_concept_ids)

# Filter the DataFrame to include only the concept IDs with incorrect number of lines
reprocess_df = df_filtered_no_duplicates[df_filtered_no_duplicates['concept_id'].isin(incorrect_line_concept_ids)]

############################################################################################################
# Might need to run several times to generate data for all medical concepts

prompt_list = []
# Regenerate and send requests for the identified concept IDs
for concept_id, group in reprocess_df.groupby('concept_id'):
    medical_concept = group['medical_concept'].iloc[0].replace("/", "_")  # Replace slashes to avoid path issues
    unique_phrases = group['informal_phrase'].unique().tolist()

    # Prepare few-shot learning prompt
    examples = unique_phrases[:10] if len(unique_phrases) >= 10 else unique_phrases
    prompt = f"Given medical concept: '{medical_concept}', generate exactly 20 informal phrases that related, don't worry about the policy. Here are some examples of informal phrase for this medical concept:\n"
    for example in examples:
        prompt += f"- {example}\n"
    prompt += "Generate new informal phrases:"
    prompt_list.append(prompt)
    # Send request
    try:
        response = send_request(prompt)
        generated_phrases = response['choices'][0]['text'].strip().split('\n')
        # Ensure exactly 20 unique phrases are generated; adjust logic as needed
        generated_phrases = list(dict.fromkeys(generated_phrases))[:20]

        output_lines = [f"{concept_id}||{medical_concept}||{phrase}" for phrase in generated_phrases]

        # Save output to a file specific to the current medical concept
        output_path = os.path.join(output_base_dir, f"{concept_id}_{medical_concept}.txt")
        with open(output_path, 'w') as file:
            for line in output_lines:
                file.write(line + '\n')
        
        print(f"Output for {medical_concept} saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred for concept {concept_id}: {e}")


############################################################################################################
# Clean the generated informal phrases

# Specify the directory containing your text files
# Should be same as output_base_dir
source_directory = 'Replace with path to the desired directory'
# Specify the directory where the cleaned files will be written
target_directory = 'Replace with path to the desired directory'

# Make sure the target directory exists, create if it doesn't
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Loop through each file in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.txt'):
        # Construct the full path to the source and target files
        source_file_path = os.path.join(source_directory, filename)
        target_file_path = os.path.join(target_directory, filename)
        
        # Open the source file and read lines
        with open(source_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        cleaned_lines = []
        
        # Process each line
        for line in lines:
            # Replace "- " with "" in each line
            line = line.replace("- ", "")
            # Then, remove the numbering (digits followed by a period and space)
            cleaned_line = re.sub(r'\d+\.\s+', '', line)
            cleaned_lines.append(cleaned_line)
        
        # Write the cleaned lines to a new file in the target directory
        with open(target_file_path, 'w', encoding='utf-8') as file:
            file.writelines(cleaned_lines)

# Specify the filename for the combined file
combined_filename = 'Replace with desired file path with specific filename.txt'

# Construct the full path for the combined file
combined_file_path = os.path.join(target_directory, combined_filename)

# Open the combined file in write mode
with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
    # Loop through each file in the target directory
    for filename in os.listdir(target_directory):
        if filename.endswith('.txt') and filename != combined_filename:
            # Construct the full path to the current file
            file_path = os.path.join(target_directory, filename)
            # Open the current file in read mode
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the content of the current file
                content = file.read()
                # Write the content to the combined file
                combined_file.write(content)

print(f"All files have been combined into {combined_filename}")
