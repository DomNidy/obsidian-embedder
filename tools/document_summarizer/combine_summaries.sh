#!/bin/bash

# This script will search for all files in the current working directory that are
# suffixed with "_summaries_only.txt". The content of all of these files will be
# converted into XML format. This is intended to be used with on the summary files
# produced by the document_summarizer.
#
# Example output format:
# <FileChunks file_name="file.md">
#     <Chunk>Some content here...</Chunk>
# </FileChunks>

if [ $# -eq 0 ]; then
    echo "Error: Output file path is required"
    echo "Usage: $0 <output_file_path>"
    exit 1
fi

output_file=$1

# Create or clear the output file and add XML header and root opening tag
echo '<?xml version="1.0" encoding="UTF-8"?>' >$output_file
echo '<root>' >>$output_file

# Find all files ending with summaries_only.txt and process them
find . -regex .*summaries_only.txt$ -print0 | while IFS= read -r -d '' file; do
    # Write opening XML tag with file name
    echo "<FileChunks file_name=\"$file\">" >>$output_file

    # Read the file line by line, escape special characters, and wrap in Chunk tags
    while IFS= read -r line; do
        if [ ! -z "$line" ]; then # Only process non-empty lines
            escaped_line=$(echo "$line" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
            echo "    <Chunk>$escaped_line</Chunk>" >>$output_file
        fi
    done <"$file"
    # Write closing XML tag
    echo "</FileChunks>" >>$output_file
done

# Add root closing tag
echo '</root>' >>$output_file
