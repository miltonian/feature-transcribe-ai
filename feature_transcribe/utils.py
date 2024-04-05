import os
import fnmatch

def parse_gitignore(directory):
    """Parse .gitignore file and return a list of patterns to exclude."""
    ignore_patterns = []
    gitignore_path = os.path.join(directory, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    ignore_patterns.append(stripped_line)
    return ignore_patterns

def is_ignored(path, ignore_patterns, root_directory):
    """Check if the given path matches any of the ignore patterns, considering the root directory."""
    relative_path = os.path.relpath(path, root_directory)
    for pattern in ignore_patterns:
        # Directories in .gitignore could be listed with a trailing slash, which fnmatch doesn't handle by default.
        # Adding a special case to handle directory patterns.
        if pattern.endswith('/'):
            pattern = pattern.rstrip('/')
            if relative_path.startswith(pattern):
                return True
        elif fnmatch.fnmatch(relative_path, pattern):
            return True
    return False

def find_files(directory, file_extensions=('.swift', '.ts', '.js', '.tsx', '.jsx', '.py', '.html', '.css', '.scss', '.prisma', '.ex', '.schema', '.vue', '.vuex'), exclude='Pods'):
    ignore_patterns = parse_gitignore(directory)
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns, directory)]
        if exclude in dirs:
            dirs.remove(exclude)
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(file_extensions) and not is_ignored(full_path, ignore_patterns, directory):
                yield full_path
