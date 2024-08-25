import json
import os
import re

with open("./inputs/class.json") as f:
    classified = json.load(f)


def extract_two(fname):
    s = re.match(r"img(\d+)-is(\d+)", fname)
    print(s, fname)
    if not s:
        return
    # return s


for file in os.listdir("./inputs"):
    x = extract_two(file)
    if x == None:
        continue

    imnum, clss = x
    classified[f"{imnum}.png"] = clss

    os.rename(f"./inputs/{file}", f"./inputs/{imnum}.png")
