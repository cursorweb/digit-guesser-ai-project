import json
import os
import re

with open("./inputs/class.json") as f:
    classified = json.load(f)


def extract_two(fname):
    s = re.match(r"img(\d+)-is(\d+)", fname)
    if not s:
        return
    return s.group(1), s.group(2)


for file in os.listdir("./inputs/data"):
    x = extract_two(file)
    if x == None:
        continue

    imnum, clss = x
    classified[f"{imnum}.png"] = clss

    os.rename(f"./inputs/data/{file}", f"./inputs/data/{imnum}.png")

with open("./inputs/class.json", "w") as f:
    json.dump(classified, f)
