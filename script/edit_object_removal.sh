#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <output_folder> <config_file> "
    exit 1
fi


output_folder="$1"
config_file="$2"

if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi



# Remove the selected object
python edit_object_removal.py -m ${output_folder} --config_file ${config_file}
