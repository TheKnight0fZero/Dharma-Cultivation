# clean_translator.py
"""
Clean the exported translator_core.py file.
Removes Jupyter-specific code that breaks regular Python.
"""

print("🧹 Cleaning translator_core.py...")

# Read the file
with open('translator_core.py', 'r') as f:
    lines = f.readlines()

# Clean lines
cleaned_lines = []
skip_next = False

for line in lines:
    # Skip Jupyter magic commands
    if 'get_ipython()' in line:
        cleaned_lines.append('# ' + line)  # Comment out instead
        continue
    
    # Skip %pip, %matplotlib, etc.
    if line.strip().startswith('%'):
        cleaned_lines.append('# ' + line)
        continue
    
    # Skip !command lines
    if line.strip().startswith('!'):
        cleaned_lines.append('# ' + line)
        continue
    
    # Keep regular lines
    cleaned_lines.append(line)

# Write cleaned version
with open('translator_core_clean.py', 'w') as f:
    f.writelines(cleaned_lines)

print("✅ Created translator_core_clean.py")
print("📝 Now update translator_integration.py to import from translator_core_clean")
