import re

# Read the content of counter.rs
with open('examples/counter.rs', 'r') as file:
    lines = file.readlines()

def position(iterable, predicate):
    for i, value in enumerate(iterable):
        if predicate(value):
            return i
    return None

# Remove comments but keep the first line of the top module comment
first_module_comment_line = None
in_module_comment = False
filtered_lines = []

for i, line in enumerate(lines):
    # Check if it's a module comment line
    if line.strip().startswith('//!'):
        if first_module_comment_line is None:
            # This is the first line of the module comment, keep it
            first_module_comment_line = line
            filtered_lines.append(line)
            in_module_comment = True
        # Skip all other module comment lines
        continue
    # Check if it's a regular comment line
    elif line.strip().startswith('//'):
        # Skip regular comment lines
        continue
    else:
        # Not a comment line, keep it
        filtered_lines.append(line)

lines = filtered_lines

mod_utils_line = position(lines, lambda line: line.startswith('mod utils'))
lines = lines[:mod_utils_line] + lines[mod_utils_line + 3:]

# Join the lines into a single string
content = ''.join(lines)

# Replace `example_plugin` with `((DefaultPlugins, JonmoPlugin))`
content = re.sub(r'examples_plugin', '(DefaultPlugins, JonmoPlugin)', content)

# Read the content of README.md
with open('README.md', 'r') as file:
    readme_content = file.read()

# Insert the content after the marker
# Define the start and end markers for the Rust code block
start_marker = '```rust,ignore'
end_marker = '```'

# Find the start and end positions of the Rust code block
start_pos = readme_content.find(start_marker)
end_pos = readme_content.find(end_marker, start_pos + len(start_marker))

# Replace the content between the markers
if start_pos != -1 and end_pos != -1:
    new_readme_content = (readme_content[:start_pos + len(start_marker)] + '\n' + content + readme_content[end_pos:])
else:
    new_readme_content = readme_content

# Write the updated content back to README.md
with open('README.md', 'w') as file:
    file.write(new_readme_content)
