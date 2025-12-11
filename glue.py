import os
import sys

# CONFIGURATION
INPUT_LIST_FILE = "context_files"
OUTPUT_FILENAME = "llm_context"

def get_file_content(file_path):
    """
    Reads a file and returns its content.
    Handles encoding errors (e.g., binary files) gracefully.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def determine_language(file_path):
    """
    Maps file extensions to markdown language tags for syntax highlighting.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.rs': 'rust',
        '.go': 'go',
        '.json': 'json',
        '.md': 'markdown',
        '.sql': 'sql',
        '.sh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.dockerfile': 'dockerfile'
    }
    return mapping.get(ext, '')

def read_paths_from_file(list_file_path):
    """
    Parses the input file list. Ignores comments (#) and empty lines.
    """
    paths = []
    if not os.path.exists(list_file_path):
        print(f"ERROR: Could not find the input file '{list_file_path}'")
        print("Please create this file and list your target file paths in it.")
        return []

    print(f"Reading paths from: {list_file_path}")
    with open(list_file_path, 'r') as f:
        for line in f:
            # Clean whitespace
            path = line.strip().strip('"').strip("'")
            
            # Skip empty lines or comments
            if not path or path.startswith("#"):
                continue
                
            paths.append(path)
    return paths

def create_llm_context_file(file_paths, output_file):
    """
    Combines files into a single context file formatted for LLMs.
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("Below is the codebase context. Each file is separated by a header and code block.\n\n")
        
        count = 0
        for path in file_paths:
            if not os.path.exists(path):
                print(f"  [Skipping] File not found: {path}")
                continue
                
            print(f"  [Processing] {path}")
            
            content = get_file_content(path)
            lang = determine_language(path)
            
            # --- FORMATTING START ---
            out.write(f"File: {path}\n")
            out.write("=" * 50 + "\n")
            out.write(f"```{lang}\n")
            out.write(content)
            
            if not content.endswith('\n'):
                out.write("\n")
                
            out.write("```\n\n")
            # --- FORMATTING END ---
            count += 1
            
    return count

def main():
    print("--- Code Context Gatherer ---")
    
    files = read_paths_from_file(INPUT_LIST_FILE)
    
    if files:
        count = create_llm_context_file(files, OUTPUT_FILENAME)
        if count > 0:
            print(f"\nSUCCESS: Combined {count} files into '{OUTPUT_FILENAME}'")
        else:
            print("\nWARNING: No valid files were processed.")
    else:
        print("No files found to process.")

if __name__ == "__main__":
    main()