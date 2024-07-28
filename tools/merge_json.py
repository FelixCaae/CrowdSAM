import sys
import json

# Check for the correct number of command-line arguments
if len(sys.argv) < 4:
    print("Usage: python merge_json.py output.json input1.json input2.json ...")
    sys.exit(1)

# The first argument is the output file
output_file = sys.argv[1]

# The remaining arguments are input JSON files
json_files = sys.argv[2:]

# Initialize an empty list to hold merged data
merged_data = []

# Load and merge JSON files
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        merged_data.extend(data)

# Write merged data to the specified output JSON file
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print("JSON files merged successfully.")