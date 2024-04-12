from typing import List, Dict, Any
from feature_transcribe.openai_api import generate_embedding
import re 
from pathlib import Path

def parse_file(file_path: str, api_key: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Generate embedding for the entire file content
    embedding = generate_embedding(content[:5000], api_key)  

    file_info = {
        "path": file_path,
        "embedding": embedding
    }

    return file_info

def parse_code(path:str, code: str, api_key: str) -> Dict[str, Any]:
    
    # Generate embedding for the entire code
    embedding = generate_embedding(code[:5000], api_key)  

    file_info = {
        "path": path,
        "embedding": embedding
    }

    return file_info

def parse_code_and_ast(ast:str, code: str, api_key: str) -> Dict[str, Any]:
    
    # Generate embedding for the entire code
    # embedding = generate_embedding(ast[:5000], api_key)  

    file_info = {
        "ast": ast,
        "code": code,
        "embedding": []
    }

    return file_info

ts_js_patterns = {
    'class': r'class\s+(\w+)',
    'interface': r'interface\s+(\w+)',
    'function': r'function\s+(\w+)|(\w+)\s*\(.*?\)\s*:\s*\w+\s*=>|\w+\s*:\s*function\s*\(.*?\)',
    'variable': r'(let|var)\s+(\w+)\s*(:\s*[^=]+)?\s*(=\s*[^;]+)?;',
    'constant': r'const\s+(\w+)\s*(:\s*[^=]+)?\s*(=\s*[^;]+)?;',
    # 'router': r'router\.(get|patch|post|put|delete)\(\s*[\'"`](.*?)[\'"`]'
    'router': r'router\.(get|post|put|delete)\(\s*[\'"`](.*?)[\'"`]'
}

def parse_ts_js_code(path: str):
    file_content = Path(path).read_text()
    file_results = {
        'classes': [],
        'interfaces': [],
        'functions': [],
        'variables': [],
        'constants': [],
        'routes': []
    }

    # Search for patterns in the file and capture names and details
    for key, pattern in ts_js_patterns.items():
        for match in re.finditer(pattern, file_content):
            if key in ['class', 'interface', 'function', 'router']:
                name = next((m for m in match.groups() if m), 'anonymous')
                if key == 'interface':
                    file_results['interfaces'].append(f'interface {name}')
                elif key == 'class':
                    file_results['classes'].append(f'class {name}')
                elif key == 'router':
                    method, path = match.groups()
                    name = f"{method.upper()} {path}"  
                    print(f'router: {name}')
                    file_results['routes'].append(name)
                else: 
                    file_results['functions'].append(f'func {name}')
            else: 
                groups = match.groups()
                var_type = groups[0]
                name = groups[1]
                type_hint = groups[2] if len(groups) > 2 and groups[2] else ''
                initial_value = groups[3] if len(groups) > 3 and groups[3] else ''
                detail = f'{var_type} {name}{type_hint}{initial_value}'
                if key == 'variable':
                    file_results['variables'].append(detail.strip())
                else:  
                    file_results['constants'].append(detail.strip())

    return file_results

def parse_ts_js_code_paths(file_paths):
    results = {}
    for file_path in file_paths:
        results[file_path] = parse_ts_js_code(file_path)

    return results

# Define patterns to match Swift constructs
swift_patterns = {
    'class': r'\bclass\s+\w+',
    'struct': r'\bstruct\s+\w+',
    'function': r'\bfunc\s+\w+',
    'variable': r'\bvar\s+\w+',
    'constant': r'\blet\s+\w+',
    # Add more patterns here if needed
}

def parse_swift_file(path):
    # Initialize a dictionary to hold parsed data
    parsed_data = {'classes': [], 'structs': [], 'functions': [], 'variables': [], 'constants': []}
    
    # Read the content of the file
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Apply each pattern and populate the parsed_data dictionary
    for key, pattern in swift_patterns.items():
        matches = re.findall(pattern, content)
        if key == 'class':
            parsed_data['classes'].extend(matches)
        elif key == 'struct':
            parsed_data['structs'].extend(matches)
        elif key == 'function':
            parsed_data['functions'].extend(matches)
        elif key == 'variable':
            parsed_data['variables'].extend(matches)
        elif key == 'constant':
            parsed_data['constants'].extend(matches)

    return parsed_data
    
# Define a function to parse Swift files
def parse_swift_files(file_paths):
    
    # Initialize a dictionary to hold parsed data
    parsed_data = {file_path: {'classes': [], 'structs': [], 'functions': [], 'variables': [], 'constants': []}
                   for file_path in file_paths}
    
    # Iterate over each file path
    for file_path in file_paths:
        parsed_data[file_path] = parse_swift_file(file_path)
        
    return parsed_data

import subprocess
def get_node_ast(path: str, search: str):
    print(path, search)
    command = ['node', 'ts-to-ast.js', path, search]
    result = subprocess.run(command, capture_output=True, text=True)
    # ast = result.stdout
    try:
        # parsed_output = json.loads(result.stdout)
        outputs = result.stdout.strip().split('\n')
        ast_snippets = []
        for output in outputs:
            try:
                # Parse the JSON output into a Python dictionary
                parsed_output = json.loads(output)
                ast_snippets.append(parsed_output)
                # print("AST %s" % parsed_output["ast"])
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                print(f"Faulty output: {output}")
                return []
        # parsed_output = json.loads(result.stdout)

        # Extract serialized AST and code into separate variables
        return ast_snippets
        # serialized_ast = parsed_output["serializedAst"]
        # code = parsed_output["code"]
        # print("AST %s" % serialized_ast)
        # return {
        #     "code": code,
        #     "ast": serialized_ast
        # }
    except json.JSONDecodeError as e:
        print("JSON decoding failed:", e)
        print("Faulty output:", result.stdout)
        return []

    

import subprocess
import json
import time

def parse_tls():
    # Start the TypeScript Language Server
    proc = subprocess.Popen(
        ['typescript-language-server', '--stdio'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send an LSP initialization request. This is a simplified example.
    # A real initialization request needs additional mandatory fields.
    init_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "processId": None,
            "rootPath": None,
            "rootUri": None,
            "capabilities": {},
        }
    })

    # Write the initialization request to the language server's stdin
    proc.stdin.write(init_request + '\n')
    proc.stdin.flush()

    # Give the server a moment to respond. In a real application, you'd
    # want to select or poll for readability.
    time.sleep(1)

    # Read the response from the language server's stdout
    response = proc.stdout.readline()
    print("Response from TypeScript Language Server: %s" % response)

    # Terminate the process
    proc.terminate()
