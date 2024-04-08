import argparse
from feature_transcribe.code_parser import parse_swift_file, parse_code, parse_ts_js_code
from feature_transcribe.utils import find_files
import json
from pathlib import Path

def main(directory: str, api_key: str, output_file: str):
    """
    Main function to parse TypeScript files, extract components, and generate embeddings.

    Parameters:
    - directory (str): Directory containing TypeScript files.
    - api_key (str): OpenAI API key for generating embeddings.
    - output_file (str): File to save the embeddings output.
    """
    
    embeddings_output = []
    for file in find_files(directory):
        print("Processing %s" % file)
        extension = file.split('.')[-1].lower()
        
        # Define extensions for TypeScript/JavaScript and related types
        ts_js_variants = {'ts', 'js', 'vue', 'jsx', 'tsx'}
        
        # Check the file type and process accordingly
        if extension in ts_js_variants:
            content = parse_ts_js_code(file)
        elif extension == 'swift':
            content = parse_swift_file(file)
        else:
            content = "Unsupported file type"
        
        code_embedding = parse_code(file, json.dumps(content), api_key)
        embeddings_output.append(code_embedding)

    with open(output_file, 'w') as file:
        json.dump(embeddings_output, file, indent=4)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files to generate embeddings.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing Swift files")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--output_file", type=str, default="embeddings_output.json", help="Output file for embeddings")
    
    args = parser.parse_args()
    main(args.directory, args.api_key, args.output_file)
