import requests
import json
import os
import time
from openai import OpenAI

assistant_file_path = '.assistant_id.json'

def get_thread_id() -> str:
    """Retrieves the thread id from a file."""
    try:
        with open(".thread_id", 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None
    
def save_thread_id(thread_id: str):
    """Saves the last thread id to a file."""
    with open(".thread_id", 'w') as file:
        file.write(thread_id)
        
def delete_thread_id():
    """Deletes the .thread_id file."""
    try:
        os.remove(".thread_id")
        # print("Thread ID file deleted successfully.")
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def read_assistant_id_from_file():
    file_path = assistant_file_path
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data.get('assistant_id')
        except (IOError, json.JSONDecodeError) as error:
            print(f"Error reading from {file_path}: {error}")
    return None

def create_and_store_new_assistant_id(model:str, api_key: str):
    client = OpenAI(api_key=api_key)
    file_path = assistant_file_path
    try:
        # instructions = "You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded files. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. when referencing files, don't reference the ids, reference the names. if you need to do a deeper analysis of the code in certain files, please do so so you can best fulfill the request. yes, you have direct access to the content of these files and can analyze the deeply enough to fit the code you provide nicely into the existing files themselves. language of the code should match the language of files given. and yes, you have direct access to the content of these files and can analyze them deeply. yes you have access to external content directly"
        # instructions = "You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded files. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. please try to use what you see in the code i pasted and try to assume everything based on the code provided. don't assume any frameworks or dependencies are used unless the code explicitly says that"

        # Initialize the chat assistant session
        assistant = client.beta.assistants.create(
            # instructions=instructions,
            name="FeatureTranscribeAI",
            tools=[{"type": "code_interpreter"}],
            model=model
        )
        assistant_id = assistant.id
        with open(file_path, 'w') as file:
            json.dump({'assistant_id': assistant_id}, file)
        return assistant_id
    except Exception as error:
        print(f"Error creating new OpenAI Assistant: {error}")
    return None


def get_system_message_content(response):
    # Assuming response is a list of message objects
    assistant_messages_with_timestamps = [(message.created_at, message.content[0].text.value)
                                           for message in response.data if message.role == 'assistant']

    # Sort the collected messages by timestamp
    assistant_messages_with_timestamps.sort(key=lambda x: x[0])

    # Extract and join the sorted message texts
    # combined_message = "\n\n".join([msg[1] for msg in assistant_messages_with_timestamps])
    combined_message = [msg[1] for msg in assistant_messages_with_timestamps][-1]
    return combined_message

def send_message_to_assistant(message, api_key, model="gpt-3.5-turbo", additional_instructions="."):
    client = OpenAI(api_key=api_key)

    assistant_id = read_assistant_id_from_file()
    thread_id = get_thread_id()

    if not assistant_id:
        print("Failed to obtain a valid Assistant ID.")
        return {"message": "Failed to obtain a valid Assistant ID."}

    if not thread_id:
        run = client.beta.threads.create_and_run(
            model=model,
            assistant_id=assistant_id,
            instructions=additional_instructions,
            thread={
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
        )
    else:
        if additional_instructions:
            # empty_thread = client.beta.threads.create()
            # thread_id = empty_thread.id
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content= message
            )
            run =  client.beta.threads.runs.create(
                thread_id=thread_id,
                model=model,
                assistant_id=assistant_id,
                instructions=additional_instructions
            )
        else:
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content= message
            )
            run =  client.beta.threads.runs.create(
                thread_id=thread_id,
                model=model,
                assistant_id=assistant_id,
                instructions=additional_instructions
            )

    # Wait for the thread run to complete, checking periodically
    attempts = 0
    max_attempts = 180
    while attempts < max_attempts and not run.completed_at:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
        attempts += 1

    # Handle the completed run
    if run.completed_at:
        # Retrieve and aggregate all "assistant" messages
        thread_messages = client.beta.threads.messages.list(run.thread_id)
        assistant_messages = get_system_message_content(thread_messages)
        if not thread_id:
            save_thread_id(run.thread_id)
        return {
            "message": assistant_messages,
            "thread_id": run.thread_id
        }
    else:
        return {"message": "The run did not complete in time."}

def send_message_to_chatgpt(prompt, api_key, model="gpt-3.5-turbo", max_tokens=1000):
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    message = completion.choices[0].message.content
    return message

def generate_embedding(input_text, api_key):
    """
    Generate an embedding for the given text using OpenAI's API.

    Parameters:
    - api_key (str): The API key for authenticating with the OpenAI API.
    - text (str): The text to generate an embedding for.

    Returns:
    - dict: The response from the OpenAI API containing the embedding.

    Throws:
    - requests.exceptions.RequestException: An error occurred making the HTTP request.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": input_text
    }
    try:
        # Make the HTTP request to the OpenAI API and return the JSON response.
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
        return response.json()['data'][0]['embedding']
    except:
        return []

