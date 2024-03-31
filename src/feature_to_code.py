import argparse
from openai_api import generate_embedding
import numpy as np
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings_with_code(file_path):
    """Load embeddings and corresponding file paths from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    embeddings = []
    paths = []
    for item in data:
        # Use .get() to safely access 'embedding', defaulting to None if not found
        embedding = item.get('embedding')
        
        # Skip the item if 'embedding' is falsey
        if not embedding:
            continue

        embeddings.append(embedding)
        paths.append(item['path'])  # Assuming 'path' key always exists
        # components.append(item['component'])  # Assuming 'component' key always exists

    return np.array(embeddings), paths

def load_code_from_paths(paths):
    """Load code content from a list of file paths."""
    codes = []
    for path in paths:
        try:
            with open(path, 'r') as file:
                codes.append(file.read())
        except FileNotFoundError:
            codes.append("File not found: " + path)
    return codes

def load_new_feature_embedding(file_path):
    """Load the new feature description and embedding from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    description = data['new_feature_description']  # Extract the description
    embedding = np.array(data['embedding'])  # Extract and convert the embedding to a numpy array
    return description, embedding

def feature_to_code_segments(embeddings, paths, new_feature_embedding, top_n=150, confidence_threshold=0.3):
    """
    Find and return the top N most relevant code segments to the new feature,
    as well as those above a certain confidence threshold.
    """

    # Calculate cosine similarity
    similarities = cosine_similarity(new_feature_embedding.reshape(1, -1), embeddings)[0]

    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    # print(similarities)
    # Filter indices by confidence threshold
    high_confidence_indices = [i for i in sorted_indices if similarities[i] >= confidence_threshold]
    
    # If there are fewer high confidence items than top_n, fill up with the highest other items until reaching top_n
    if len(high_confidence_indices) < top_n:
        top_indices = sorted_indices[:top_n]
    else:
        # Ensure we don't miss top N due to confidence filtering
        top_indices = high_confidence_indices[:max(len(high_confidence_indices), top_n)]

    # Load the code for the relevant paths
    codes = load_code_from_paths([paths[i] for i in top_indices])

    # Select the corresponding codes, paths, and similarities
    relevant_codes_paths = [(codes[idx], paths[i], similarities[i]) for idx, i in enumerate(top_indices)]

    # Sort by similarity score in descending order
    relevant_codes_paths.sort(key=lambda x: x[2], reverse=True)

    return relevant_codes_paths

def aggregate_code_segments(relevant_code_paths):
    """Aggregate selected code segments into a single coherent context."""
    aggregated_code = '\n\n'.join([code for code, _, _ in relevant_code_paths])
    return aggregated_code

# TODO: write something about different model usage
def make_openai_request(prompt, aggregated_code, api_key, model="gpt-3.5-turbo", max_tokens=150000):
    """
    Send a chat request to the OpenAI API using aggregated code as context.
    
    :param aggregated_code: The aggregated code segments as a single string, serving as context.
    :param model: The model identifier.
    :param max_tokens: Maximum number of tokens to generate.
    :return: The API response as a string.
    """ 
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You're a coding assistant. Below is some code related to a feature in development. Please respond with the code to make this new feature request or bug fix"},
            {"role": "user", "content": aggregated_code},
            {"role": "user", "content": prompt},
        ],
        # "temperature": 0.5,
        # "max_tokens": max_tokens,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    print(response)
    result = response.json()
    print(result)
    return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files to generate embeddings.")
    parser.add_argument('--feature', type=str, required=True, help='Description of the new feature')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    args = parser.parse_args()

    prompt = args.feature
    api_key = args.api_key
    new_feature_embedding = generate_embedding(prompt, api_key)
    print(new_feature_embedding)
    new_feature_embedding = np.array(new_feature_embedding)

    # Load the embeddings and paths from another JSON (as per your existing structure)
    embeddings, paths = load_embeddings_with_code('embeddings_output.json')
    
    # Find the top N most relevant code segments
    relevant_code_paths = feature_to_code_segments(embeddings, paths, new_feature_embedding, top_n=15)
    
    # Aggregate the selected code segments
    aggregated_code = aggregate_code_segments(relevant_code_paths)
    
    # Now, use the prompt and aggregated_code with your existing function
    response_text = make_openai_request(prompt, aggregated_code, api_key)
    print("OpenAI API Chat Response:")
    print(response_text)