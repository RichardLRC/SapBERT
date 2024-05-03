import openai
import pandas as pd
import os
import time

# Setup OpenAI credentials
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

AskAPatient = pd.read_csv("./AskAPatient_Concept.csv")
AskAPatient_formal_concept = AskAPatient["Medical_concept"]

for i, current_formal_concept in enumerate(AskAPatient_formal_concept):
    #start_phrase = "Please generate 20 informal phrases from social text that include synonyms or similar meaning words for the medical concept " + current_formal_concept + "."
    start_phrase = "Please generate 20 informal phrases from social text, each in a specific context or scenario, that can be mapped to the medical concept " + current_formal_concept + "."
    try:
        response = send_request(start_phrase)
        text = response['choices'][0]['text'].strip()
        lines = text.split('\n')
        cur_column = pd.DataFrame(lines, columns=[current_formal_concept])
        current_formal_concept_correct_version = current_formal_concept.replace("/", "_")
        cur_column.to_csv("Replace with path to the desired directory  " + current_formal_concept_correct_version + ".csv", index = False)
    except Exception as e:
        print(f"An error occurred for concept {current_formal_concept}: {e}") 
    