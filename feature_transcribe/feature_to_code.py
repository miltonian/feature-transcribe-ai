import argparse
import codecs
from feature_transcribe.openai_api import generate_embedding, send_message_to_chatgpt, send_message_to_assistant, create_and_store_new_assistant_id, read_assistant_id_from_file, get_thread_id, save_thread_id
import sys
import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from IPython.display import Markdown
import textwrap
from feature_transcribe.code_parser import parse_swift_file, parse_code, parse_ts_js_code, get_node_ast, parse_code_and_ast
from feature_transcribe.diff_utils import read_existing_embeddings



def load_embeddings_with_code(file_path: str, include_tests: bool=False):
    """Load embeddings and corresponding file paths from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    embeddings = []
    codes = []
    summaries = []
    paths = []
    for item in data:
        embedding = item.get('embedding')

        if not embedding:
            continue
        
        if not include_tests and "test." in item["path"] or ".spec." in item["path"]:
            continue

        embeddings.append(embedding)
        codes.append(item['code'])  
        summaries.append(item['summary'])  
        paths.append(item['path'])  

    return np.array(embeddings), codes, summaries, paths

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

def find_relevance(embeddings, paths, asts, codes, new_feature_embedding, top_n=30):
    # Calculate cosine similarity
    similarities = cosine_similarity(new_feature_embedding.reshape(1, -1), embeddings)[0]
   
    # Calculate mean and standard deviation of similarities
    mean_similarity = np.mean(similarities)
    std_dev_similarity = np.std(similarities)

    # Adjust the threshold more selectively based on distribution
    dynamic_confidence_threshold = mean_similarity + (0 * std_dev_similarity)

    # Filter indices by dynamic threshold
    high_confidence_indices = [i for i, similarity in enumerate(similarities) if similarity >= dynamic_confidence_threshold]

    # Sort high-confidence indices by similarity, then select top N
    relevant_indices = sorted(high_confidence_indices, key=lambda i: similarities[i], reverse=True)[:top_n]
    print("relevant indices")
    print(relevant_indices)
    return [paths[i] for i in relevant_indices]
    
def feature_to_code_segments(embeddings, paths, summaries, codes, new_feature_embedding, top_n=15):
    # Calculate cosine similarity
    similarities = cosine_similarity(new_feature_embedding.reshape(1, -1), embeddings)[0]

    # Calculate mean and standard deviation of similarities
    mean_similarity = np.mean(similarities)
    std_dev_similarity = np.std(similarities)

    # Adjust the threshold more selectively based on distribution
    dynamic_confidence_threshold = mean_similarity + (1.5 * std_dev_similarity)

    # Filter indices by dynamic threshold
    high_confidence_indices = [i for i, similarity in enumerate(similarities) ]#if similarity >= dynamic_confidence_threshold]

    # Sort high-confidence indices by similarity, then select top N

    relevant_indices = sorted(high_confidence_indices, key=lambda i: similarities[i], reverse=True)[:top_n]

    # Extract code for relevant paths
    # codes = load_code_from_paths([asts[i] for i in relevant_indices])

    # Compile relevant codes and paths, ensuring to sort by similarity
    relevant_codes_paths = [(summaries[idx], paths[i], similarities[i]) for idx, i in enumerate(relevant_indices)]
    relevant_codes_paths.sort(key=lambda x: x[2], reverse=True)

    # relevant_codes_paths.sort(key=itemgetter(1)) # sort by path

    # # Use groupby
    # relevant_codes_paths = {key: list(group) for key, group in groupby(relevant_codes_paths, key=itemgetter(0))}

    
    # print(relevant_codes_paths)

    for _, path, similarity in relevant_codes_paths:
        print(f"path: {path}, Confidence: {similarity:.2f}")

    return relevant_codes_paths

def aggregate_code_segments(relevant_code_paths):
    """Aggregate selected code segments into a single coherent context."""
    aggregated_code = '\n\n'.join([code for code, _, _ in relevant_code_paths])
    return aggregated_code

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import re

def main(paths_to_modify: str, code_identifiers_to_modify: str, prompt: str, api_key: str, model: str, top_n=15):
    """
    Main function to generate embedding for the feature, 

    Parameters:
    - feature (str): Feature request / issue.
    - api_key (str): OpenAI API key for generating embeddings.
    - model (str): OpenAI model used for generating embeddings.
    """
    
    
    
    # path_to_modify = "/tasks/api/routes/properties.test.ts"
    # code_identifier_to_modify = "CallExpression: describe(\"PUT /properties/:propertyId\""

    # Load the embeddings and paths from another JSON (as per your existing structure)
    embeddings, codes, summaries, paths = load_embeddings_with_code('embeddings_output.json', "test" in prompt)
    
    new_feature_embedding = generate_embedding(prompt, api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    # Find the top N most relevant code segments
    relevant_code_paths = feature_to_code_segments(embeddings, paths, summaries, codes, new_feature_embedding, top_n)
    summaries = []
    paths = []
    files_and_summaries = []
    for summary, path, confidence in relevant_code_paths:
        summaries.append(summary)
        print(f"SUMMARY: {summary}")
        paths.append(path)
        files_and_summaries.append(f"file {path} summary: {summary}")
    
    
    
    assistant_id = read_assistant_id_from_file()
    is_new_conversation = False
    if not assistant_id:
        is_new_conversation = True
        create_and_store_new_assistant_id(model, api_key)
    
    files_and_summaries_str = "\n".join(files_and_summaries)
    # assumed_relevant_files_message = send_message_to_chatgpt(f"""
    assumed_relevant_files_message = send_message_to_assistant(f"""
        Which of these files are most likely where the code for the feature request would be? or would it need a new file and where? explain why. your response should be json formatted like this {{"answer": "your response about my question", "paths": ["path/to/file1", "path/to/file2"], "codeSnippets": ["code 1", "code 2"]}}, codeSnippets should be the exact copy of anything that resembles a code snippet in the user input
        {files_and_summaries_str}

        FEATURE DESCRIPTION: {prompt}
    """, api_key, model, "your response should be json")
    assumed_relevant_files_message = assumed_relevant_files_message['message']
    
    json_string = assumed_relevant_files_message.replace("```json", "").replace("```", "")
    json_response = json.loads(json_string)
    paths_to_modify_arr = [item.strip() for item in paths_to_modify.split(',')]
    # if not paths_to_modify_arr:
    paths_to_modify_arr = json_response['paths']
    code_identifiers_to_modify_arr = json_response['codeSnippets']
    assumed_relevant_files_message = json_response['answer']
    
    
    temp_paths_to_modify_arr = []
    for path in paths:
        for path_2 in paths_to_modify_arr:
            if path_2 in path:
                temp_paths_to_modify_arr.append(path)

    paths_to_modify_arr = temp_paths_to_modify_arr
    code_for_context = ""
    for path_to_modify in paths_to_modify_arr:
        path = path_to_modify
        for code_identifier_to_modify in code_identifiers_to_modify_arr:
            existing_embeddings = get_node_ast(path, code_identifier_to_modify)
            search_value = code_identifier_to_modify
            pattern = r"(^Identifier: (\w+)|StringLiteral: \"([^\"]+)\")"

            filtered_by_path = [item for item in existing_embeddings if path in item["path"]]
            for data in filtered_by_path:
                matches = get_node_ast(data['path'], search_value)
                found_value = None
                if matches:
                    found_value = matches[0]
                    if found_value:
                        data = found_value
                        ast = found_value['ast']
                        if ast:
                            code = data['code']
                            ast = data['ast']
                            path = data['path']
                            code_for_context += f"file: {path}"
                            pattern = r"Identifier: (\w+)"
                            identifiers = re.findall(pattern, ast)
                            # print(f"identifiers: {identifiers}")
                            # include imports
                            for identifier in identifiers:
                                filtered_by_path_2 = [item for item in filtered_by_path if re.search(pattern, item['ast'])]
                                find_import_pattern = fr"ImportDeclaration: .*{identifier}"
                                found_import = [item for item in filtered_by_path_2 if re.search(find_import_pattern, item['ast'])]
                                if found_import:
                                    import_code = found_import[0].get('code', None)
                                    code_for_context += import_code
                                    code_for_context += "\n"

                            # include main code
                            code_for_context += "\n\n"
                            code_for_context += code
                            code_for_context += "\n\n"        

                            for identifier in identifiers:
                                filtered_by_path_2 = [item for item in filtered_by_path if re.search(pattern, item['ast'])]
                                find_import_pattern = fr"ImportDeclaration: .*{identifier}"
                                found_import = [item for item in filtered_by_path_2 if re.search(find_import_pattern, item['ast'])]
                                if found_import:
                                    found_path = found_import[0].get('path', None)
                                    found_import = found_import[0].get('ast', None)
                                    # print(f"found_import {found_import}")

                                    # Pattern to find the value following "Identifier: "
                                    identifier_pattern = r"Identifier: ([^\n]+)"

                                    # Pattern to find the value following "StringLiteral: "
                                    string_literal_pattern = r"StringLiteral: \"([^\"]+)\""
                                    # Using re.search since we're interested in the first occurrence
                                    identifier_match = re.search(identifier_pattern, found_import)

                                    # Extracting the matched groups if found
                                    identifier_value = identifier_match.group(1) if identifier_match else None
                                    string_literal_matches = re.findall(string_literal_pattern, found_import)
                                    string_literal_value = string_literal_matches[-1] if string_literal_matches else None

                                    print("found Identifier:", identifier_value)
                                    print("found StringLiteral:", string_literal_value)

                                    # existing_embeddings_2 = read_existing_embeddings("embeddings_output.json")
                                    filtered_by_import_path = get_node_ast(data['path'], string_literal_value.replace('.', '').replace('@', ''))
                                    # content_list = get_node_ast(found_path, "")
                                    # for content in content_list:
                                    #     ast = content['ast']
                                    #     code = content['code']
                                    #     print("content %s" % content)
                                    #     code_embedding = parse_code_and_ast(ast, code, api_key)
                                    #     code_embedding['path'] = found_path
                                    #     existing_embeddings_2.append(code_embedding)
                                    # filtered_by_import_path = [item for item in existing_embeddings_2 if (string_literal_value.replace('.', '').replace('@', '')) in item["path"]]
                                    # The specific value you're looking for after "Identifier: "
                                    search_value = identifier_value

                                    # Regular expression pattern to find "Identifier: <value>"
                                    pattern = r"Identifier: (\w+)"

                                    for item in filtered_by_import_path:
                                        ast = item.get("ast", "")
                                        
                                        # Search for the pattern
                                        match = re.search(pattern, ast)
                                        # found = None
                                        if match:
                                            # Extract the value found after "Identifier: "
                                            found_value = match.group(1)
                                            
                                            # Check if the found value equals your search value
                                            if found_value == search_value:
                                                # found = match
                                                print("The value after 'Identifier: ' equals", search_value)
                                                print(item['code'])
                                                code_for_context += f"file: {item['path']}"
                                                code_for_context += "\n\n"
                                                code_for_context += item['code']
                                                code_for_context += "\n\n"
    
                            

    if not code_for_context:
        if is_new_conversation: 
            return {
                "relevant_code_paths": files_and_summaries,
                "response": assumed_relevant_files_message
            }
        else:
            response = send_message_to_assistant(
                prompt, 
                api_key, 
                model
            )
            return {
                'response': response['message']
            }
            
    improved_feature_prompt = send_message_to_assistant(
        f"""
            feature prompt: {prompt}
        """, 
        api_key, 
        model,
        """
            improve the feature prompt the user lists in the request to be as if coming from a product person directly into the hands of a software engineer to develop this feature in code entirely and, for improving the prompt, please analyze the uploaded files. you should provide the developer with technical specifications/guidelines, and break down the feature into a series of basic steps so the developer can tackle one at a time

            Respond with the improved prompt and nothing else.
        """
    )
    new_feature_embedding = generate_embedding(improved_feature_prompt['message'], api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    response_text = send_message_to_assistant(
        f"""
            first, analyze the code from the request and derive the programming language used from the code and then perform the following in that language:
            
            {prompt}
            
            EXISTING CODE: {code_for_context}
        """, 
        api_key, 
        model,
        "You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded files. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. please try to use what you see in the code i pasted and try to assume everything based on the code provided. don't assume any frameworks or dependencies are used unless the code explicitly says that"
    )
    
    return {
        "relevant_code_paths": [code_for_context],
        "response": response_text['message']
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