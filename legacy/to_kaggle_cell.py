import argparse
import sys

S = """
def write_to_file(name, s):
    with open(name, "w") as io:
        io.write(s)
"""


list_of_files = [
    "accuracy_index.py",
    "models.py",
    "datasets.py",
    "loss_functions.py",
    "config.py",
    "train.py",
] + sys.argv[1:]

for file in list_of_files:
    s = None
    basename = file.split("/")[-1]
    with open(file) as io:
        s = ("").join(io.readlines())
        s = f"""\
write_to_file("{basename}", \"\"\"\\
{s}\
\"\"\")
        """
    S = f"{S}\n{s}"

print(S)
