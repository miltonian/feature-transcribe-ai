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
    # Calculate cosine similarity
    similarities = cosine_similarity(new_feature_embedding.reshape(1, -1), embeddings)[0]

    # Calculate mean and standard deviation of similarities
    mean_similarity = np.mean(similarities)
    std_dev_similarity = np.std(similarities)

    # Adjust the threshold more selectively based on distribution
    dynamic_confidence_threshold = mean_similarity + (1.5 * std_dev_similarity)

    # Filter indices by dynamic threshold
    high_confidence_indices = [i for i, similarity in enumerate(similarities) if similarity >= dynamic_confidence_threshold]

    # Sort high-confidence indices by similarity, then select top N
    relevant_indices = sorted(high_confidence_indices, key=lambda i: similarities[i], reverse=True)[:top_n]

    # Extract code for relevant paths
    codes = load_code_from_paths([paths[i] for i in relevant_indices])

    # Compile relevant codes and paths, ensuring to sort by similarity
    relevant_codes_paths = [(codes[idx], paths[i], similarities[i]) for idx, i in enumerate(relevant_indices)]
    relevant_codes_paths.sort(key=lambda x: x[2], reverse=True)

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

def make_openai_request(prompt, paths, api_key, model="gpt-3.5-turbo", max_tokens=1000):
    client = OpenAI(api_key=api_key)

    # Initialize a list to store file IDs
    file_ids = []
    file_names = []

    # Iterate over relevant code paths and upload each as a file
    for code_path in paths:
        with open(code_path, "rb") as file:
            try:
                # Attempt to upload the file and store the resulting file ID
                upload_response = client.files.create(file=file, purpose='assistants')
                file_ids.append(upload_response.id)
                file_name_after_last_slash = code_path.split('/')[-1]
                file_names.append(file_name_after_last_slash)
            except Exception as e:
                print(f"Failed to upload {code_path}: {e}")

    # Check if any files were uploaded successfully
    if not file_ids:
        return "No files were uploaded successfully."

    file_names_string = ",".join(file_names)

    # Clear and concise instructions
    instructions = "Please provide a coding solution based on the context provided in the uploaded files. Base Your response on the file ids included in the request. your response should include the code i need to copy and paste into my application to solve the request"

    # Explicitly mention the use of uploaded files in your prompt
    prompt_with_reference = "of the files uploaded, " + file_names_string + "analyze the files you think are most relevant first, then after you analyze, complete the following request and tell me how to add it to my codebase within these files: " + prompt

    # Initialize the chat assistant session
    assistant = client.beta.assistants.create(
        # instructions="You are a software engineer. Based on the files in your knowledge, please use them as context and respond with the code to make this new feature request or bug fix given the files . Please include as much code as you can to complete the feature request or bug fix. Don't include any files or links in your response.",
        instructions=instructions,
        name="FeatureTranscribeAI",
        tools=[{"type": "code_interpreter"}],
        model=model,
        # file_ids=file_ids
    )
    
    run = client.beta.threads.create_and_run(
        model=model,
        assistant_id=assistant.id,
        thread={
            "messages": [
                {
                    "role": "user",
                    "content": prompt_with_reference,
                    "file_ids": file_ids 
                }
            ]
        }
    )

    # Wait for the thread run to complete, checking periodically
    attempts = 0
    max_attempts = 120
    while attempts < max_attempts and not run.completed_at:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
        attempts += 1

    # Handle the completed run
    if run.completed_at:
        # Retrieve and aggregate all "assistant" messages
        thread_messages = client.beta.threads.messages.list(run.thread_id)
        assistant_messages = get_system_message_content(thread_messages)
        return assistant_messages
    else:
        return "The run did not complete in time."

def make_openai_request_for_prompt(prompt, api_key, model="gpt-3.5-turbo", max_tokens=1000):
    client = OpenAI(api_key=api_key)

    # Clear and concise instructions
    instructions = f"""
    heres the feature description i'm giving openai along with the relevant files. give me a better prompt from this that uses the following strategies below. only return the prompt for me. nothing else.
        - Increase Detail in Prompts: Provide more detailed prompts that focus on the specific aspect of the code or logic you're interested in. This can help guide the model's attention to the relevant parts of the files.
        - Break Down the Problem: Instead of asking for an analysis of the entire file or a large block of code, break down your request into smaller, more manageable questions that focus on specific functions, methods, or logic flows.
        - Provide Context: Along with the code, include explanations or comments within your prompt that highlight the key areas of interest or specific functionalities you need help with. This can help the model better understand the context and provide more targeted insights.
        - Iterative Querying: Use an iterative approach where you start with a general inquiry and then drill down based on the model's responses. This can help uncover more detailed and specific insights over multiple interactions.
        
        and below is the feature description i want you to improve:
        
        {prompt}
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instructions}
        ]
    )
    message = completion.choices[0].message.content
    print("message")
    print(message)
    return message

def get_system_message_content(response):
    # Assuming response is a list of message objects
    assistant_messages_with_timestamps = [(message.created_at, message.content[0].text.value)
                                           for message in response.data if message.role == 'assistant']

    # Sort the collected messages by timestamp
    assistant_messages_with_timestamps.sort(key=lambda x: x[0])

    # Extract and join the sorted message texts
    combined_message = "\n\n".join([msg[1] for msg in assistant_messages_with_timestamps])
    return combined_message


# def get_system_message_content(response):
#     # Initialize a list to collect messages along with their creation timestamps
#     assistant_messages_with_timestamps = []

#     # Iterate through each message in the response data
#     for message in response.data:
#         # Check if the message role is 'assistant'
#         if message.role == 'assistant':
#             # Extract the timestamp and content for each message
#             timestamp = message.created_at
#             content_blocks = message.content  # This should be a list of content blocks
#             for block in content_blocks:
#                 if hasattr(block, 'text') and hasattr(block.text, 'value'):
#                     # Append a tuple of (timestamp, message text) for each block
#                     assistant_messages_with_timestamps.append((timestamp, block.text.value))
    
#     # Sort the collected messages by timestamp (the first item in each tuple)
#     assistant_messages_with_timestamps.sort(key=lambda x: x[0])

#     # Extract and join the sorted message texts into a single string with line breaks
#     combined_message = "\n\n".join([msg[1] for msg in assistant_messages_with_timestamps])
#     return combined_message


def main(prompt: str, api_key: str, model: str):
    """
    Main function to generate embedding for the feature, 

    Parameters:
    - feature (str): Feature request / issue.
    - api_key (str): OpenAI API key for generating embeddings.
    - model (str): OpenAI model used for generating embeddings.
    """
    # prompt = """
    # Given the detailed feature description and the steps to reproduce the issue, here is an optimized prompt to guide OpenAI in analyzing the provided code and offering specific insights or solutions:

    # "I'm working on a feature within the 'ralph' environment that involves updating property details through a PUT request to internalUrl/v2/properties. The issue arises with the identity_verification_report_images field, which should only accept 'selfie' or 'document' as valid inputs. However, it currently accepts any text input, leading to unexpected behavior in subsequent processes, specifically when interacting with external/identity/applicants/${applicantId}/images, where it still returns a 200 response along with the selfie image even for invalid inputs.

    # Here are the steps to reproduce the issue:

    #     Make a PUT request to internalUrl/v2/properties and update the identity_verification_report_images field with an invalid value like 'helloWorld'.
    #     Then, make a request to externalUrl/identity/applicants/${applicantId}/images.

    # The expected behavior is that the system should reject any values for identity_verification_report_images other than 'selfie' or 'document'. If invalid values are set, it should not proceed to return a 200 response from the external/identity/applicants/${applicantId}/images endpoint.

    # Given this context and the files provided, which include handlers and validations for these fields and endpoints, can you identify where the validation for identity_verification_report_images should be implemented within the provided files? Additionally, please suggest a specific code change or implementation that would enforce the correct validation, ensuring that only 'selfie' or 'document' are accepted as valid inputs for the identity_verification_report_images field. Also, consider how to handle cases where invalid inputs are provided, such as returning an appropriate error response."

    # This prompt leverages the strategies of providing detailed context, focusing on the specific problem, and asking for a targeted solution, which should help in obtaining a more precise and useful response from OpenAI.
    # """
    
    new_feature_embedding = generate_embedding(prompt, api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    # Load the embeddings and paths from another JSON (as per your existing structure)
    embeddings, paths = load_embeddings_with_code('embeddings_output.json')

    # Find the top N most relevant code segments
    relevant_code_paths = feature_to_code_segments(embeddings, paths, new_feature_embedding, top_n=10)

    # Aggregate the selected code segments
    # aggregated_code = aggregate_code_segments(relevant_code_paths)

    # Now, use the prompt and aggregated_code with your existing function
    onlypaths = [path for _, path, _ in relevant_code_paths]

    relevant_code_paths_with_confidence = [f"Path: {path}, Confidence: {confidence:.2f}" for _, path, confidence in relevant_code_paths]
    
    prompt_response = make_openai_request_for_prompt(prompt, api_key, model)
    response_text = make_openai_request(prompt_response, onlypaths, api_key, model)

    return {
        'relevant_code_paths': relevant_code_paths_with_confidence,
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

