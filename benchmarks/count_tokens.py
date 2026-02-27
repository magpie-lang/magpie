#!/usr/bin/env python3
"""Token counter for benchmark files using tiktoken (cl100k_base / GPT-4 tokenizer)."""
import sys
import json
import os

def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_approx(text)

def count_tokens_approx(text: str) -> int:
    """Approximate token count: ~4 chars per token for code."""
    import re
    tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
    return len(tokens)

def analyze_file(path: str) -> dict:
    with open(path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    non_empty = [l for l in lines if l.strip()]
    comment_lines = [l for l in lines if l.strip().startswith('//') or l.strip().startswith(';;')]

    return {
        "file": os.path.basename(path),
        "bytes": len(content.encode('utf-8')),
        "characters": len(content),
        "lines_total": len(lines),
        "lines_code": len(non_empty) - len(comment_lines),
        "lines_comment": len(comment_lines),
        "lines_blank": len(lines) - len(non_empty),
        "tokens": count_tokens_tiktoken(content),
    }

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("Usage: count_tokens.py file1 file2 ...")
        sys.exit(1)

    results = []
    for f in files:
        r = analyze_file(f)
        results.append(r)

    print(json.dumps(results, indent=2))
