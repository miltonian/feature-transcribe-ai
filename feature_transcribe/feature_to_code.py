import argparse
from feature_transcribe.openai_api import generate_embedding
import numpy as np
import json
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI



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

def feature_to_code_segments(embeddings, paths, new_feature_embedding, top_n=15):
    """
    Find and return the top N most relevant code segments to the new feature,
    with a more stringent dynamic confidence threshold.
    """

    # Calculate cosine similarity
    similarities = cosine_similarity(new_feature_embedding.reshape(1, -1), embeddings)[0]

    # Dynamically adjust the confidence threshold to be more stringent
    mean_similarity = np.mean(similarities)
    std_dev_similarity = np.std(similarities)
    # Multiply std_dev by a factor (e.g., 1.5) to make the threshold more selective
    dynamic_confidence_threshold = mean_similarity + (1.5 * std_dev_similarity)

    # Sort indices by similarity in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Select indices with similarities above the more stringent threshold
    high_confidence_indices = [i for i in sorted_indices if similarities[i] >= dynamic_confidence_threshold]

    # Strictly enforce the top_n limit, only considering high-confidence indices up to top_n
    relevant_indices = high_confidence_indices[:top_n]

    # Load the code for the relevant paths
    codes = load_code_from_paths([paths[i] for i in relevant_indices])

    # Compile relevant codes, paths, and similarities, ensuring to sort by similarity
    relevant_codes_paths = [(codes[idx], paths[i], similarities[i]) for idx, i in enumerate(relevant_indices)]
    relevant_codes_paths.sort(key=lambda x: x[2], reverse=True)

    # Optionally, print paths and their corresponding confidence
    for _, path, similarity in relevant_codes_paths:
        print(f"Path: {path}, Confidence: {similarity:.2f}")

    return relevant_codes_paths



def aggregate_code_segments(relevant_code_paths):
    """Aggregate selected code segments into a single coherent context."""
    aggregated_code = '\n\n'.join([code for code, _, _ in relevant_code_paths])
    return aggregated_code

# def make_openai_request(prompt, aggregated_code, api_key, model="gpt-3.5-turbo", max_tokens=150000):
#     """
#     Send a chat request to the OpenAI API using aggregated code as context.

#     :param aggregated_code: The aggregated code segments as a single string, serving as context.
#     :param model: The model identifier.
#     :param max_tokens: Maximum number of tokens to generate.
#     :return: The API response as a string.
#     """ 
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
#     data = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": "You're a coding assistant. Below is some code related to a feature in development. Please respond with the code to make this new feature request or bug fix"},
#             {"role": "user", "content": aggregated_code},
#             {"role": "user", "content": prompt},
#         ],
#         # "temperature": 0.5,
#         # "max_tokens": max_tokens,
#     }

#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
#     print(response)
#     result = response.json()
#     print(result)
#     return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

def make_openai_request(prompt, aggregated_code, api_key, model="gpt-3.5-turbo", max_tokens=1000):
    """
    Creates a chat assistant session, uploads aggregated code as a file to OpenAI, 
    and sends a chat message referencing the file ID to get a response.
    
    :param aggregated_code: The aggregated code segments as a single string, serving as context.
    :param model: The model identifier.
    :param max_tokens: Maximum number of tokens to generate, within API limits.
    :return: The API response as a string.
    """
    client = OpenAI(api_key=api_key)

    # Save the aggregated code to a temporary file
    temp_file_path = "temp_aggregated_code.txt"
    with open(temp_file_path, "w") as file:
        file.write(aggregated_code)
    
    # Upload the file to OpenAI and get the file ID
    with open(temp_file_path, 'rb') as file:
        upload_response = client.files.create(file=file, purpose='assistants')
        # file_id = upload_response['id']
        file_id = upload_response.id
    
    # Initialize the chat assistant session
    assistant = client.beta.assistants.create(
        instructions="You are a software engineer. Based on the files uploaded in the users request, please use them as context and respond with the code to make this new feature request or bug fix. Please include as much code as you can to complete the feature request or bug fix. Don't include any files or links in your response.",
        name="FeatureTranscribeAI",
        tools=[{"type": "code_interpreter"}],
        model=model
    )

    run = client.beta.threads.create_and_run(
        assistant_id=assistant.id,
        thread={
            "messages": [
            {"role": "user", "file_ids": [file_id], "content": prompt}
            ]
        }
    )
    # Initialize attempt counter
    attempts = 0
    max_attempts = 60

    # Loop until run is completed or max_attempts reached
    while attempts < max_attempts:
        # Retrieve the current state of the run
        run = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        print(f"Attempt {attempts + 1}: Run state - Completed: {bool(run.completed_at)}")

        # Check if run.completed_at is truthy, indicating completion
        if run.completed_at:
            print("Run completed.")
            break  # Exit the loop if run is completed

        # Wait for 1 second before the next attempt
        time.sleep(1)
        attempts += 1  # Increment the attempt counter

    # Optional: Handle the case when max_attempts are reached but run is not completed
    if not run.completed_at:
        print("Max attempts reached. The run did not complete in time.")
        return "Max attempts reached. The run did not complete in time."

    thread_messages = client.beta.threads.messages.list(run.thread_id)
    system_message_content = get_system_message_content(thread_messages)
    print(system_message_content)
    return system_message_content.strip()
    print(thread_messages)

    # Assuming you want to return some part of the run response
    # Adjust the return statement according to what you need from the run object
    print(run)
    return run.responses[-1].content if run.completed_at else "Run did not complete."
    

    # Send a message to the assistant, including the file_id in the system message for context
    # chat_response = assistant.send_messages(messages=[
    #     # {"role": "system", "content": f"The following is a code context: file={file_id}"},
    #     # {"role": "system", "content": f"A file with ID {file_id} contains some code related to a feature in development. "},
    #     {"role": "user", "file_ids": [file_id], "content": prompt}
    # ])
    
    # # Extract the response content
    # response_content = chat_response['responses'][0]['message']['content']

    return "" # response_content.strip()

def get_system_message_content(response):
    # Initialize a list to collect messages along with their creation timestamps
    assistant_messages_with_timestamps = []

    # Iterate through each message in the response data
    for message in response.data:
        # Check if the message role is 'assistant'
        if message.role == 'assistant':
            # Extract the timestamp and content for each message
            timestamp = message.created_at
            content_blocks = message.content  # This should be a list of content blocks
            for block in content_blocks:
                if hasattr(block, 'text') and hasattr(block.text, 'value'):
                    # Append a tuple of (timestamp, message text) for each block
                    assistant_messages_with_timestamps.append((timestamp, block.text.value))
    
    # Sort the collected messages by timestamp (the first item in each tuple)
    assistant_messages_with_timestamps.sort(key=lambda x: x[0])

    # Extract and join the sorted message texts into a single string with line breaks
    combined_message = "\n\n".join([msg[1] for msg in assistant_messages_with_timestamps])
    return combined_message


def main(prompt: str, api_key: str, model: str):
    """
    Main function to generate embedding for the feature, 

    Parameters:
    - feature (str): Feature request / issue.
    - api_key (str): OpenAI API key for generating embeddings.
    - model (str): OpenAI model used for generating embeddings.
    """

    new_feature_embedding = generate_embedding(prompt, api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    # Load the embeddings and paths from another JSON (as per your existing structure)
    embeddings, paths = load_embeddings_with_code('embeddings_output.json')

    # Find the top N most relevant code segments
    relevant_code_paths = feature_to_code_segments(embeddings, paths, new_feature_embedding, top_n=7)

    # Aggregate the selected code segments
    aggregated_code = aggregate_code_segments(relevant_code_paths)

    # Now, use the prompt and aggregated_code with your existing function
    response_text = make_openai_request(prompt, aggregated_code, api_key, model)

    return {
        'relevant_code_paths': [path for _, path, _ in relevant_code_paths],
        'response': response_text
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files to generate embeddings.")
    parser.add_argument('--feature', type=str, required=True, help='Description of the new feature')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='OpenAI model')
    args = parser.parse_args()

    prompt = args.feature
    api_key = args.api_key
    model = args.model

    main(prompt, api_key, model)

