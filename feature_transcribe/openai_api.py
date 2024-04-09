import requests

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

