files = [
    "dataset/chunk_3.txt",
    "dataset/chunk_4.txt",
    "dataset/chunk_5.txt"
]

output_file = "merged_chunks.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in files:
        with open(fname, "r", encoding="utf-8") as infile:
            outfile.write(infile.read().strip() + "\n\n")
print(f"Merged into {output_file}")
