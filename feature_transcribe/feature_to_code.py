import argparse
from feature_transcribe.openai_api import generate_embedding
import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import google.generativeai as genai
import os
from IPython.display import Markdown
import textwrap
from feature_transcribe.code_parser import parse_swift_file, parse_code, parse_ts_js_code, get_node_ast, parse_code_and_ast

def read_assistant_id_from_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data.get('assistant_id')
        except (IOError, json.JSONDecodeError) as error:
            print(f"Error reading from {file_path}: {error}")
    return None

def create_and_store_new_assistant_id(file_path: str, model:str, api_key: str):
    client = OpenAI(api_key=api_key)
    try:
        # instructions = "You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded files. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. when referencing files, don't reference the ids, reference the names. if you need to do a deeper analysis of the code in certain files, please do so so you can best fulfill the request. yes, you have direct access to the content of these files and can analyze the deeply enough to fit the code you provide nicely into the existing files themselves. language of the code should match the language of files given. and yes, you have direct access to the content of these files and can analyze them deeply. yes you have access to external content directly"
        instructions = "You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded files. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. please try to use what you see in the code i pasted and try to assume everything based on the code provided. don't assume any frameworks or dependencies are used unless the code explicitly says that"

        # Initialize the chat assistant session
        assistant = client.beta.assistants.create(
            instructions=instructions,
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

def load_embeddings_with_code(file_path):
    """Load embeddings and corresponding file paths from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    embeddings = []
    codes = []
    asts = []
    paths = []
    for item in data:
        embedding = item.get('embedding')

        if not embedding:
            continue
        
        # if "test." in item["path"] or ".spec." in item["path"]:
        #     continue

        embeddings.append(embedding)
        codes.append(item['code'])  
        asts.append(item['ast'])  
        paths.append(item['path'])  

    return np.array(embeddings), codes, asts, paths

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
    
from itertools import groupby
from operator import itemgetter
def feature_to_code_segments(embeddings, paths, asts, codes, new_feature_embedding, top_n=15):
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
    relevant_codes_paths = [(codes[idx], paths[i], similarities[i]) for idx, i in enumerate(relevant_indices)]
    relevant_codes_paths.sort(key=lambda x: x[2], reverse=True)

    relevant_codes_paths.sort(key=itemgetter(1)) # sort by path

    # # Use groupby
    # relevant_codes_paths = {key: list(group) for key, group in groupby(relevant_codes_paths, key=itemgetter(0))}

    
    print(relevant_codes_paths)

    # for _, ast, similarity in relevant_codes_paths:
    #     print(f"AST: {ast}, Confidence: {similarity:.2f}")

    return relevant_codes_paths

def aggregate_code_segments(relevant_code_paths):
    """Aggregate selected code segments into a single coherent context."""
    aggregated_code = '\n\n'.join([code for code, _, _ in relevant_code_paths])
    return aggregated_code

def upload_files(paths, api_key): 
    client = OpenAI(api_key=api_key)

    # Initialize a list to store file IDs
    file_ids = []
    file_names = []
    print(paths)
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

    if not file_ids:
        return "No files were uploaded successfully."

    return {
        'file_ids': file_ids, 
        'file_names': file_names
    }
        
def send_message_to_assistant(assistant_id, message, file_ids, api_key, thread_id, model="gpt-3.5-turbo", max_tokens=1000, additional_instructions=None):
    client = OpenAI(api_key=api_key)
    
    if thread_id:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content= message
        )
        run =  client.beta.threads.runs.create(
            thread_id=thread_id,
            model=model,
            assistant_id=assistant_id
        )
    elif additional_instructions:
        empty_thread = client.beta.threads.create()
        thread_id = empty_thread.id
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            file_ids=file_ids,
            role="user",
            content= message
        )
        run =  client.beta.threads.runs.create(
            thread_id=thread_id,
            model=model,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions
        )
    else: 
        run = client.beta.threads.create_and_run(
            model=model,
            assistant_id=assistant_id,
            thread={
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                        "file_ids": file_ids 
                    }
                ]
            }
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
        return {
            "message": assistant_messages,
            "thread_id": run.thread_id
        }
    else:
        return "The run did not complete in time."

def make_openai_request_for_prompt(prompt, api_key, model="gpt-3.5-turbo", max_tokens=1000, override_instructions=None):
    client = OpenAI(api_key=api_key)

    # Clear and concise instructions
    instructions = f"""
        I'm enhancing my project with a new feature and need targeted coding assistance for seamless integration.  Translate this feature description into a prompt for chatgpt using open ai's prompt documentation. your response should include only the prompt i can copy and paste
        
        Feature Description:
        
        {prompt}
    """
    if override_instructions:
        instructions = override_instructions
    

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

def delete_file(file_id, api_key):
    client = OpenAI(api_key=api_key)

    content = client.files.delete(file_id)
    return content
    
def get_system_message_content(response):
    # Assuming response is a list of message objects
    assistant_messages_with_timestamps = [(message.created_at, message.content[0].text.value)
                                           for message in response.data if message.role == 'assistant']

    # Sort the collected messages by timestamp
    assistant_messages_with_timestamps.sort(key=lambda x: x[0])

    # Extract and join the sorted message texts
    combined_message = "\n\n".join([msg[1] for msg in assistant_messages_with_timestamps])
    return combined_message

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def prompt_gemini(prompt: str, paths, api_key, model):
    api_key="AIzaSyC53YxgGDam8_r1UqhGP9QGgMGLk8DTCGw"
    codes = load_code_from_paths(paths)
    code = "\n".join(codes)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"""
        You are a principal software engineer. Please provide the entire coding solution based on the context provided in the uploaded code. Base your response on the files included in the request. your response should include all of the exact code i need to copy and paste into my application to solve the request. please do so so you can best fulfill the request. yes, you have direct access to the content of these files and can analyze the deeply enough to fit the code you provide nicely into the existing code: 
        {prompt}
        
        And here is my existing code:
        {code}
    """)
    
    print("--GEMINI RESPONSE--")
    print(response.text)
    
    return response.text 
    
def extract_python_code_blocks(input_string):
    start_marker = "```python"
    end_marker = "```"
    code_blocks = []
    
    start_pos = input_string.find(start_marker)
    while start_pos != -1:
        # Adjust start position to get the actual start of the code
        adjusted_start_pos = start_pos + len(start_marker)
        # Find the end marker from the adjusted start position
        end_pos = input_string.find(end_marker, adjusted_start_pos)
        
        # If the end marker is found, extract the code block
        if end_pos != -1:
            code_block = input_string[adjusted_start_pos:end_pos].strip()
            code_blocks.append(code_block)
            # Move past this block for the next iteration
            start_pos = input_string.find(start_marker, end_pos + len(end_marker))
        else:
            # If no end marker is found, stop the loop
            break
    
    return code_blocks


# Import necessary libraries
import re
from pathlib import Path


# Define a function to extract the code block of a given component from the file content
def extract_code_block(content, component_name):

    # Search for the component declaration
    start_index = content.find(component_name)
    if start_index == -1:
        return None  # Component not found

    # Find the opening brace of the component
    start_brace_index = content.find('{', start_index)
    if start_brace_index == -1:
        return None  # Opening brace not found, unusual case

    # Initialize brace count and iterate through the content to find the matching closing brace
    brace_count = 1
    for i in range(start_brace_index + 1, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            
        if brace_count == 0:
            return content[start_index:i + 1]  # Return the component code block
    return None  # Matching closing brace not found

def extract_relevant_code(path, parsed_data):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Initialize a dictionary to hold the extracted code for this file
        extracted_code = {'structs': [], 'classes': [], 'interfaces': [], 'functions': [], 'variables': [], 'constants': [], 'routes': []}

        # Check if file_path exists in parsed_data before proceeding
        # if path in parsed_data:
        file_data = parsed_data # [path]  # Reference to the data for this file
        
        for class_name in file_data.get('classes', []):
            class_code = extract_code_block(content, class_name)
            if class_code:
                extracted_code['classes'].append(class_code)

        for interface_name in file_data.get('interfaces', []):
            interface_code = extract_code_block(content, interface_name)
            if interface_code:
                extracted_code['interfaces'].append(interface_code)

        for function_name in file_data.get('functions', []):
            function_code = extract_code_block(content, function_name)
            if function_code:
                extracted_code['functions'].append(function_code)

        for variable_name in file_data.get('variables', []):
            variable_code = extract_code_block(content, variable_name)
            if variable_code:
                extracted_code['variables'].append(variable_code)

        for constant_name in file_data.get('constants', []):
            constant_code = extract_code_block(content, constant_name)
            if constant_code:
                extracted_code['constants'].append(constant_code)

        for struct_name in file_data.get('structs', []):
            struct_code = extract_code_block(content, struct_name)
            if struct_code:
                extracted_code['structs'].append(struct_code)

        for struct_name in file_data.get('routes', []):
            route_code = extract_code_block(content, struct_name)
            if route_code:
                extracted_code['routes'].append(route_code)
        # else:
        #     print(f"Warning: No parsed data available for {path}")
            
        return extracted_code
    
def extract_relevant_code_paths(file_paths, parsed_data):
    extracted_code = {}

    for file_path in file_paths:
        extracted_code[file_path] =  extract_relevant_code(file_path, parsed_data)


    return extracted_code

def prompt_open_ai(prompt: str, codes, api_key, model):
    
    # files = upload_files(codes, api_key)
    # file_ids = files['file_ids']
    # file_names = files['file_names']

    # prompt_response = make_openai_request_for_prompt(prompt, api_key, model)

    file_path = 'assistant_id.json'
    assistant_id = read_assistant_id_from_file(file_path)

    if not assistant_id:
        assistant_id = create_and_store_new_assistant_id(file_path, model, api_key)

    if assistant_id:
        # file_names_string = ",".join(file_names)
        
        prompt1 = f"""
            feature prompt: {prompt}
        """
        # prompt1 = f"""
        #     Identify and list all files within the application related to the [Feature Description] request. Exclude any files not directly contributing to the functionality described in the feature request. Provide a list categorized by 'Relevant' and 'Not Relevant' based on their direct contribution to the feature, without detailing the reasons for classification. Just include the file names under each category.

        #     Feature description:  {prompt}
        # """
        # prompt1 = f"""
        #     Analyze the provided code or files to identify the key components and patterns common to the programming languages represented. Generate a parser script capable of extracting a wide array of code components from files in these languages. The script should be able to recognize and process components such as but not limited to classes, functions, methods, variable declarations, imports, and comments. Determine the most suitable scripting language for the parser based on your analysis of the code files' contents and overall structure. The parser should output the extracted information in a structured JSON format, adaptable for use with an embeddings API. Tailor the output format to ensure it captures the essence and functionality of each code component, including name, type, and relevant details or code snippets. IMPORTANT: your response should be the parsing code only so i can immediately execute on it.

        #     replace any paths like (path/to/your/file) to [[path_to_your_file]]
        # """
        # prompt1 = f"""
        #     Analyze the uploaded files to identify sections relevant to implementing a new feature that [feature description]. Consider elements like [list of keywords/patterns identified that are specific to the programming language detected in these files]. The code is written in [language]. Highlight any parts of the code that may need to be changed or extended to accommodate this feature.

        #     What parts of the code are relevant, and what changes might be needed?
            
        #     replace the variables wrapped in brackets with what you deem necessary to answer the question above
        #     feature description: {prompt}

        # """
        # STEP 1: break down the request into actionable steps to take 
        response_text_obj = send_message_to_assistant(assistant_id, prompt1, [], api_key, None, model, 1000, """
            improve the feature prompt the user lists in the request to be as if coming from a product person directly into the hands of a software engineer to develop this feature in code entirely and, for improving the prompt, please analyze the uploaded files. you should provide the developer with technical specifications/guidelines, and break down the feature into a series of basic steps so the developer can tackle one at a time

            Respond with the improved prompt and nothing else.
        """)
        response_text = response_text_obj['message']
        thread_id = response_text_obj['thread_id']
        print("--PROMPT1 RESPONSE--")
        print(response_text)
        print(thread_id)

    else:
        print("Failed to obtain a valid Assistant ID.")
        
    return response_text

def prompt_open_ai2(prompt: str, code: str, api_key: str, model):

    file_path = 'assistant_id.json'
    assistant_id = read_assistant_id_from_file(file_path)

    if not assistant_id:
        assistant_id = create_and_store_new_assistant_id(file_path, model, api_key)

    if assistant_id:
        
        prompt1 = f"""
            first, analyze the code from the request and derive the programming language used from the code and then perform the following in that language:
            
            {prompt}
            
            EXISTING CODE: {code}
        """
        
        # STEP 1: break down the request into actionable steps to take 
        response_text_obj = send_message_to_assistant(assistant_id, prompt1, [], api_key, None, model)
        response_text = response_text_obj['message']
        thread_id = response_text_obj['thread_id']
        print("--PROMPT1 RESPONSE--")
        print(response_text)
        print(thread_id)

    else:
        print("Failed to obtain a valid Assistant ID.")
        
    return response_text

def main(prompt: str, api_key: str, model: str, top_n=15):
    """
    Main function to generate embedding for the feature, 

    Parameters:
    - feature (str): Feature request / issue.
    - api_key (str): OpenAI API key for generating embeddings.
    - model (str): OpenAI model used for generating embeddings.
    """

    # Load the embeddings and paths from another JSON (as per your existing structure)
    embeddings, codes, asts, paths = load_embeddings_with_code('embeddings_output.json')
    
    new_feature_embedding = generate_embedding(prompt, api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    # Find the top N most relevant code segments
    relevant_code_paths = feature_to_code_segments(embeddings, paths, asts, codes, new_feature_embedding, top_n)
    # asts = [ast for _, ast, _ in relevant_code_paths]
    
    improved_feature_prompt = prompt_open_ai(prompt, codes, api_key, model)
    new_feature_embedding = generate_embedding(improved_feature_prompt, api_key)
    new_feature_embedding = np.array(new_feature_embedding)

    # relevant_code_paths_with_confidence = [f"Code: {code}, Confidence: {confidence:.2f}" for _, code, confidence in relevant_code_paths]

    embeddings_output = []

    # for ast in asts:
    # for i in range(1, len(asts)):
        # code_groupings_def = parse_swift_files(paths)
        # extension = path.split('.')[-1].lower()
            
        # # Define extensions for TypeScript/JavaScript and related types
        # ts_js_variants = {'ts', 'js', 'vue', 'jsx', 'tsx'}
        # # Check the file type and process accordingly
        # if extension in ts_js_variants:
        #     # code_groupings_def = parse_ts_js_code(path)
        #     result = get_node_ast(path)
        #     extracted_code = result['code']
        #     ast = result['ast']
        # elif extension == 'swift':
        #     code_groupings_def = parse_swift_file(path)
        # else:
        #     code_groupings_def = "Unsupported file type"

        # extracted_code = extract_relevant_code(path, code_groupings_def)
        # result = get_node_ast(path)
        # extracted_code = result['code']
        # ast = result['ast']
        # START HERE
        # ast = asts[i]
        # extracted_code = codes[i]

        # embedding = parse_code_and_ast(ast, extracted_code, api_key)
        # embeddings_output.append(embedding)
        # END HERE

        # interfaces = extracted_code.get("interfaces", [])
        # classes = extracted_code.get("classes", [])
        # variables = extracted_code.get("variables", [])
        # constants = extracted_code.get("constants", [])
        # structs = extracted_code.get("structs", [])
        # routes = extracted_code.get("routes", [])

        # for interfaces in interfaces:
        #     embedding = parse_code(interfaces, interfaces, api_key)
        #     embeddings_output.append(embedding)
        # for classes in classes:
        #     embedding = parse_code(classes, classes, api_key)
        #     embeddings_output.append(embedding)
        # for variables in variables:
        #     embedding = parse_code(variables, variables, api_key)
        #     embeddings_output.append(embedding)
        # for constants in constants:
        #     embedding = parse_code(constants, constants, api_key)
        #     embeddings_output.append(embedding)
        # for structs in structs:
        #     embedding = parse_code(structs, structs, api_key)
        #     embeddings_output.append(embedding)
        # for routes in routes:
        #     embedding = parse_code(routes, routes, api_key)
        #     embeddings_output.append(embedding)
        
    # with open("code_embeddings_output.json", 'w') as file:
    #     json.dump(embeddings_output, file, indent=4)  

    # # Load the embeddings and paths from another JSON (as per your existing structure)
    # embeddings, codes, asts = load_embeddings_with_code('code_embeddings_output.json')
    print("EMBEDDINGS")
    print(embeddings)
    # print("PATHS")
    # print(paths)
    # Find the top N most relevant code segments
    # relevant_code = find_relevance(embeddings, paths, asts, codes, new_feature_embedding, top_n)

    # response_text = prompt_open_ai2(improved_feature_prompt, relevant_code, api_key, model)
    
    # response_text = make_openai_request_for_prompt(response_text, api_key, "gpt-3.5-turbo", 1000, "format this in html: " + response_text)
    
    # relevant_code_paths_with_confidence = ['\n\n'.join(relevant_code)]
    # relevant_code_paths_with_confidence = [f"{path}: {code}, Confidence: {confidence:.2f}" for code, path, confidence in relevant_code_paths]
    # Initialize an empty list to store your formatted strings
    relevant_code_paths_with_confidence = []

    # Initialize a variable to keep track of the last path processed
    last_path = None

    for code, path, confidence in relevant_code_paths:
        print("PATH COMPARE")
        print(path, last_path)
        if path != last_path:
            # If the current path is different from the last one, include the path in the string
            formatted_string = f"{path}: {code}, Confidence: {confidence:.2f}"
        else:
            # If the current path is the same as the last one, exclude the path from the string
            formatted_string = f"{code}, Confidence: {confidence:.2f}"
        
        # Add the formatted string to your list
        relevant_code_paths_with_confidence.append(formatted_string)
        
        # Update the last path processed
        last_path = path
    # response_text = prompt_gemini(prompt, paths, api_key, model)

    return {
        'relevant_code_paths': relevant_code_paths_with_confidence,
        'response': "response_text"
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