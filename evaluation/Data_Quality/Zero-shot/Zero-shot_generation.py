import openai
import pandas as pd
import time
import os

openai.api_key = "###########################"
openai.api_base = "###########################" 
openai.api_type = '###########################'
openai.api_version = '###########################' 
deployment_name = '###########################' 
print(openai.__version__)

# Function to send completion request
def send_request(prompt, attempts=3, timeout=900):  # Increase timeout as per need
    for attempt in range(attempts):
        try:
            response = openai.Completion.create(
                engine=deployment_name,
                prompt=prompt,
                max_tokens=3000,
                temperature=0.9,
                top_p=0.8,
                timeout=timeout  # Set timeout for each request
            )
            return response
        except openai.error.Timeout as e:
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt < attempts - 1:
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                raise  # Raise exception if all attempts fail

AskAPatient = pd.read_csv("Data_Quality/Hypothesis_three_context_enrichment/AskAPatient_Concept.csv/AskAPatient_Concept.csv")
AskAPatient_formal_concept = AskAPatient["formal_concept"]

for i, current_formal_concept in enumerate(AskAPatient_formal_concept):
    start_phrase =  "Please generate 100 informal phrase from social text which can be mapped to the medical concept " + current_formal_concept + " ." 
    try:
        response = send_request(start_phrase)
        text = response['choices'][0]['text'].strip()
        lines = text.split('\n')
        cur_column = pd.DataFrame(lines, columns=[current_formal_concept])
        cur_column.to_csv("Replace with path to the desired directory " + current_formal_concept + ".csv", index = False)
    except Exception as e:
        print(f"An error occurred for concept {current_formal_concept}: {e}")


# This is the single document generation 
# current_formal_concept = "Pain in lower limb"
# start_phrase =  "Please generate 100 informal phrase from social text which can be mapped to the medical concept " + current_formal_concept + " ." 

# try:
#     response = send_request(start_phrase)
#     text = response['choices'][0]['text'].strip()
#     lines = text.split('\n')
#     cur_column = pd.DataFrame(lines, columns=[current_formal_concept])
#     current_formal_concept_correct_version = current_formal_concept.replace("/", "_")
#     cur_column.to_csv("Replace with path to the desired directory " + current_formal_concept + ".csv", index = False)
# except Exception as e:
#     print(f"An error occurred for concept {current_formal_concept}: {e}") 
        
        

def check_file(file_path):
    """
    Check if the file has exactly 100 rows and only 1 column.
    Return the base name of the file if it does not meet the criteria.
    """
    try:
        df = pd.read_csv(file_path)
        if df.shape[0] < 90 and df.shape[1] == 1:
            base_name = os.path.basename(file_path)
            name = base_name.replace("Informal Expression for ", "").replace(".csv", "")
            return name
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def check_files_in_directory(directory):
    """
    Check all CSV files in the given directory.
    Return a list of file names that do not meet the criteria.
    """
    incorrect_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            result = check_file(file_path)
            if result:
                incorrect_files.append(result)
    return incorrect_files

# Should be the same path as the informal phrases files generated in the previous step
directory_path = "Replace with path to the desired directory" 
files_not_meeting_criteria = check_files_in_directory(directory_path)
print(files_not_meeting_criteria)

#
for i, current_formal_concept in enumerate(files_not_meeting_criteria):
    start_phrase =  "Please generate 100 informal phrase from social text which can be mapped to the medical concept " + current_formal_concept + " ." 
    try:
        response = send_request(start_phrase)
        text = response['choices'][0]['text'].strip()
        lines = text.split('\n')
        cur_column = pd.DataFrame(lines, columns=[current_formal_concept])
        cur_column.to_csv("Replace with path to the desired directory " + current_formal_concept + ".csv", index = False)
    except Exception as e:
        print(f"An error occurred for concept {current_formal_concept}: {e}")