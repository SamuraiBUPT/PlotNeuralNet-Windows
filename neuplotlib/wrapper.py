from typing import List

def generate_tex(arch: List[str], output_dir="file.tex"):
    with open(output_dir, "w") as f: 
        for c in arch:
            # print(f"Writing: {c} to file: {output_dir}")
            f.write(c)