import re

with open("Final_Project_Report.tex", "r") as f:
    lines = f.readlines()

out_lines = []
indent_level = 0
indent_str = "    "

environments_to_indent = [
    "titlepage", "justifying", "itemize", "enumerate", "figure", "thebibliography", "center"
]

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Check for \end{} to decrease indent before printing
    is_end = False
    for env in environments_to_indent:
        if stripped.startswith(f"\\end{{{env}}}"):
            indent_level = max(0, indent_level - 1)
            is_end = True
            break
            
    # Remove existing leading whitespace but keep newlines
    if not stripped:
        out_lines.append("\n")
        continue
    
    # Calculate current indent
    current_indent = indent_str * indent_level
    
    # Check if this line is an item (could be further indented for readability)
    if stripped.startswith("\\item") or stripped.startswith("\\bibitem"):
        out_lines.append(current_indent + stripped + "\n")
    else:
        out_lines.append(current_indent + stripped + "\n")
    
    # Check for \begin{} to increase indent after printing
    for env in environments_to_indent:
        if stripped.startswith(f"\\begin{{{env}}}"):
            indent_level += 1
            break

with open("Final_Project_Report.tex", "w") as f:
    f.writelines(out_lines)

