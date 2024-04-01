from typing import List, Dict, Any
from feature_transcribe.openai_api import generate_embedding

def parse_file(file_path: str, api_key: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Generate embedding for the entire file content
    embedding = generate_embedding(content[:5000], api_key)  # Adjust content slice as needed

    file_info = {
        "path": file_path,
        "embedding": embedding
    }

    return file_info
