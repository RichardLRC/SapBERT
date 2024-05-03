import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, phrases):
        self.phrases = phrases
    
    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self, idx):
        return self.phrases[idx]

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

# Load the data
# Please replace the file paths with the paths to your data files

df_original = pd.read_csv('Path to original dataset')
df_gpt = pd.read_csv('Path to GPT dataset with Zero-shot')
df_gpt_few_shot = pd.read_csv('Path to GPT dataset with Few-shot')

# Calculate the embeddings for all phrases in the dataset
# This function will return a dictionary with the phrase as the key and the embedding as the value
def get_embeddings_for_all_phrases(df):
    unique_phrases = df['informal_phrase'].unique().tolist()
    dataset = TextDataset(unique_phrases)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    embeddings_dict = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Embeddings"):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for phrase, embedding in zip(batch, batch_embeddings):
                embeddings_dict[phrase] = embedding
    return embeddings_dict

# Embeddings for the original dataset
embeddings_dict_original = get_embeddings_for_all_phrases(df_original)

# Embeddings for the GPT dataset with Zero-shot
embeddings_dict_gpt = get_embeddings_for_all_phrases(df_gpt)

# Embeddings for the GPT dataset with Few-shot
embeddings_dict_gpt_few_shot = get_embeddings_for_all_phrases(df_gpt_few_shot)


# Bootstrapping for similarity calculation
def bootstrap_similarity(embeddings_dict, df, n_iterations=5000, batch_size=64):
    np.random.seed(42)  # for reproducibility
    bootstrap_means = []
    medical_concepts = df["medical_concept"].unique()
    
    with tqdm(total=n_iterations, desc="Bootstrapping iterations") as pbar:
        for _ in range(n_iterations):
            iteration_means = []  # Store the mean similarity for all concepts in this iteration
            for _ in medical_concepts:  # Iterate over all medical concepts
                sampled_concept = np.random.choice(medical_concepts, size=1, replace=True)
                concept_phrases = df[df['medical_concept'] == sampled_concept[0]]['informal_phrase'].tolist()
                embeddings = np.array([embeddings_dict.get(phrase) for phrase in concept_phrases if phrase in embeddings_dict])
                
                # Calculate the cosine similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Get the upper triangle of the similarity matrix
                upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
                similarity_scores = similarity_matrix[upper_triangle_indices]
                 
                # Calculate the mean similarity for this concept
                iteration_means.append(np.mean(similarity_scores))  
            
            # Calculate the mean similarity for all concepts in this iteration
            bootstrap_means.append(np.mean(iteration_means)) 
            pbar.update(1)
        print(len(bootstrap_means))
        
    return bootstrap_means

# Normal distribution and plot
def test_normality_and_plot(bootstrap_means, dataset_name, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_means, kde=True)
    plt.title(f'{dataset_name} Bootstrapping Distribution')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()


# Compare the distributions of two datasets
def compare_distributions(bootstrap_means_original, bootstrap_means_gpt, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_means_original, color="blue", label="Original", kde=True, stat="count", linewidth=0)
    sns.histplot(bootstrap_means_gpt, color="red", label="GPT", kde=True, stat="count", linewidth=0)
    plt.title('Comparison of Original vs GPT Bootstrapping Means')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Calculate the similarity between the embeddings of the original and GPT datasets
bootstrap_means_original = bootstrap_similarity(embeddings_dict_original, df_original)
bootstrap_means_gpt = bootstrap_similarity(embeddings_dict_gpt, df_gpt)

# Test for normality and plot the distribution for the original and GPT datasets
test_normality_and_plot(bootstrap_means_original, "Original Dataset", "Path to save the plot")
test_normality_and_plot(bootstrap_means_gpt, "Generated Dataset", "Path to save the plot")

# Compare the distributions of the original and GPT datasets
compare_distributions(bootstrap_means_original, bootstrap_means_gpt, "Path to save the plot")


# Bootstrapping similarity calculation for the Combined dataset
def compare_concept_similarity(embeddings_dict_original, embeddings_dict_gpt, df_original, df_gpt, n_iterations=5000):
    np.random.seed(38)  # for reproducibility
    bootstrap_means = []
    common_concepts = set(df_original['medical_concept']).intersection(df_gpt['medical_concept'])
    
    with tqdm(total=n_iterations, desc="Cross-Dataset Similarity Bootstrapping") as pbar:
        for _ in range(n_iterations):
            sampled_concepts = np.random.choice(list(common_concepts), size=len(common_concepts), replace=True)
            iteration_similarities = []
            for concept in sampled_concepts:
                original_phrases = df_original[df_original['medical_concept'] == concept]['informal_phrase'].tolist()
                gpt_phrases = df_gpt[df_gpt['medical_concept'] == concept]['informal_phrase'].tolist()
                embeddings_original = np.array([embeddings_dict_original[phrase] for phrase in original_phrases if phrase in embeddings_dict_original])
                embeddings_gpt = np.array([embeddings_dict_gpt[phrase] for phrase in gpt_phrases if phrase in embeddings_dict_gpt])
 
                if embeddings_original.size == 0 or embeddings_gpt.size == 0:
                    continue
                
                similarity_matrix = cosine_similarity(embeddings_original, embeddings_gpt)
                iteration_similarities.append(np.mean(similarity_matrix))
            
            if iteration_similarities:
                bootstrap_means.append(np.mean(iteration_similarities))
            pbar.update(1)
    
    return bootstrap_means, common_concepts


bootstrap_means_cross, commone_concept = compare_concept_similarity(embeddings_dict_original, embeddings_dict_gpt, df_original, df_gpt)
test_normality_and_plot(bootstrap_means_cross, "Combine Dataset", "Path to save the plot")
print(commone_concept)

# Aggregate three distributions. Original, GPT, Combine
def compare_all_distributions(bootstrap_means_original, bootstrap_means_gpt, bootstrap_means_cross, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_means_original, color="blue", label="Original", kde=True, stat="count", linewidth=0)
    sns.histplot(bootstrap_means_gpt, color="red", label="GPT", kde=True, stat="count", linewidth=0)
    sns.histplot(bootstrap_means_cross, color="green", label="Combine-Dataset", kde=True, stat="count", linewidth=0)
    plt.title('Comparison of All Distributions')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

compare_all_distributions(bootstrap_means_original, bootstrap_means_gpt, bootstrap_means_cross, "Path to save the plot")


# Calculate the combined similarity for original and GPT Few-shot datasets
bootstrap_means_cross_few_shot, common_concepts_few_shot = compare_concept_similarity(
    embeddings_dict_original, embeddings_dict_gpt_few_shot, df_original, df_gpt_few_shot
)

# Test for normality and plot the distribution for df_gpt_few_shot cross similarity
test_normality_and_plot(
    bootstrap_means_cross_few_shot,
    "GPT Few-Shot Cross Similarity",
    "Path to save the plot"
)

# Now, compare the bootstrap_means_cross for both df_gpt and df_gpt_few_shot
def compare_cross_distributions(bootstrap_means_cross, bootstrap_means_cross_few_shot, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_means_original, color="blue", label="Original", kde=True, stat="count", linewidth=0)
    sns.histplot(bootstrap_means_cross, color="purple", label="GPT Cross", kde=True, stat="count", linewidth=0)
    sns.histplot(bootstrap_means_cross_few_shot, color="green", label="GPT Few-Shot Cross", kde=True, stat="count", linewidth=0)
    plt.title('Comparison of Cross Dataset Similarity')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right', prop={'size': 6})
    plt.savefig(save_path)
    plt.close()

# Call the function to compare cross distributions
compare_cross_distributions(
    bootstrap_means_cross,
    bootstrap_means_cross_few_shot,
    "Path to save the plot"
)


# Calculate the mean and standard deviation for the original dataset
Mo = np.mean(bootstrap_means_original)
Sdo = np.std(bootstrap_means_original)

# Calculate the mean and standard deviation for the GPT dataset
Mg = np.mean(bootstrap_means_gpt)
Sdg = np.std(bootstrap_means_gpt)

# Calculate the mean and standard deviation for the combined dataset 
Mcombine = np.mean(bootstrap_means_cross)
Sdcombine = np.std(bootstrap_means_cross)

# Calculate the mean and standard deviation for the GPT Few-shot dataset
Mfew_shot = np.mean(bootstrap_means_cross_few_shot)
Sdfew_shot = np.std(bootstrap_means_cross_few_shot)


print(f"Mo (Original Mean): {Mo:.4f}")
print(f"Sdo (Original Std Dev): {Sdo:.4f}")
print(f"Mg (Generated Mean): {Mg:.4f}")
print(f"Sdg (Generated Std Dev): {Sdg:.4f}")
print(f"Mcombine (Combine Mean): {Mcombine:.4f}")
print(f"Sdcombine (Combine Std Dev): {Sdcombine:.4f}")
print(f"Mfew_shot (Few Shot Mean): {Mfew_shot:.4f}")
print(f"Sdfew_shot (Few Shot Std Dev): {Sdfew_shot:.4f}")
