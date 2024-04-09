import subprocess
import os
from pathlib import Path
import json

def read_existing_embeddings(output_file: str) -> list:
    try:
        with open(output_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    
def remove_old_embeddings_for_changed_files(embeddings: list, changed_files: list) -> list:
    return [embedding for embedding in embeddings if embedding['path'] not in changed_files]
    
def save_last_processed_commit(commit_hash: str, hash_file: str):
    """Saves the last processed commit hash to a file."""
    with open(hash_file, 'w') as file:
        file.write(commit_hash)

def get_last_processed_commit(hash_file: str) -> str:
    """Retrieves the last processed commit hash from a file."""
    try:
        with open(hash_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def get_changed_files(directory: str, from_commit: str, to_commit: str = 'HEAD') -> list:
    """Returns a list of files that have changed between two commits."""
    command = ['git', 'diff', '--name-only', from_commit, to_commit] if from_commit else ['git', 'ls-files', '--others', '--exclude-standard']
    changed_files = subprocess.check_output(command, cwd=directory, universal_newlines=True)
    return changed_files.strip().split('\n')

def get_untracked_files(directory: str) -> list:
    """Returns a list of untracked (new) files in the given directory."""
    # Get a list of untracked files
    untracked_files = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=directory, universal_newlines=True)
    
    # Split the output by lines to get a list of file paths
    file_list = untracked_files.strip().split('\n')
    
    return file_list
