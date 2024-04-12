import argparse
from feature_transcribe.code_parser import parse_swift_file, parse_code, parse_ts_js_code, get_node_ast, parse_tls, parse_code_and_ast
from feature_transcribe.utils import find_files
import json
from pathlib import Path
from feature_transcribe.diff_utils import get_changed_files, save_last_processed_commit, get_last_processed_commit, read_existing_embeddings, remove_old_embeddings_for_changed_files, get_untracked_files
import subprocess 
from feature_transcribe.openai_api import generate_embedding, send_message_to_chatgpt
from openai import OpenAI



def main(directory: str, api_key: str, output_file: str):
    """
    Main function to parse TypeScript files, extract components, and generate embeddings.

    Parameters:
    - directory (str): Directory containing TypeScript files.
    - api_key (str): OpenAI API key for generating embeddings.
    - output_file (str): File to save the embeddings output.
    """
    
    commit_hash_file = ".last_processed_commit"
    last_commit = get_last_processed_commit(commit_hash_file)
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, universal_newlines=True).strip()
    
    changed_files = get_changed_files(directory, last_commit)
    untracked_files = get_untracked_files(directory)

    changed_files.extend(untracked_files)
    print("changed files: %s" % changed_files)
    # Read the existing embeddings and remove old entries for changed files
    existing_embeddings = read_existing_embeddings(output_file)
    updated_embeddings = remove_old_embeddings_for_changed_files(existing_embeddings, changed_files)
    # i=0
    # embeddings_output = []
    for file in find_files(directory):
        
        if last_commit and file not in changed_files:
            print("Skipping unchanged file %s" % file)
            continue
        
        print("Processing %s" % file)
        extension = file.split('.')[-1].lower()
        
        # Define extensions for TypeScript/JavaScript and related types
        ts_js_variants = {'ts', 'js', 'vue', 'jsx', 'tsx'}
        
        # Check the file type and process accordingly
        if extension in ts_js_variants:
            content = ""
            with open(file, 'r', encoding='utf-8') as file_content:
                code = file_content.read()
            if not code:
                continue

            # summary = send_message_to_chatgpt(f"""
            #     give me a brief summary of this file. include whatever you think would be important for semantic search. only return the summary in your response
            summary = send_message_to_chatgpt(f"""
                give me a very brief comma separated list of the key points of this file as if i were a product person. i just want to know what it's meant for. your response will be used for generating an embedding to help a product person to search through

                {code[:1000]}
            """, api_key, "gpt-3.5-turbo")
            embedding = generate_embedding(f"""
                filename: {file}
                {summary}
            """, api_key)  
            code_embedding = {
                'embedding': embedding,
                'path': file,
                'summary': summary,
                'code': code
            }
            updated_embeddings.append(code_embedding)
            # content_list = get_node_ast(file)
            # for content in content_list:
            #     ast = content['ast']
            #     code = content['code']
            #     print("content %s" % content)
                # code_embedding = parse_code_and_ast(ast, code, api_key)
            #     code_embedding['path'] = file
            #     updated_embeddings.append(code_embedding)
            # i+=1
            # if i==5:
            #     break
                
        elif extension == 'swift':
            content = parse_swift_file(file)
            print("content %s" % content)
            code_embedding = parse_code_and_ast(ast, json.dumps(content), api_key)
            code_embedding['path'] = file
            updated_embeddings.append(code_embedding)
        else:
            content = "Unsupported file type"
        # i+=1
    with open(output_file, 'w') as file:
        json.dump(updated_embeddings, file, indent=4)  

    save_last_processed_commit(current_commit, commit_hash_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files to generate embeddings.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing Swift files")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--output_file", type=str, default="embeddings_output.json", help="Output file for embeddings")
    
    args = parser.parse_args()
    main(args.directory, args.api_key, args.output_file)
