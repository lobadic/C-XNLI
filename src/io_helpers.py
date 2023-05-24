import json


def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as infile:
        return json.load(infile)


def save_json(filepath: str, data):
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile)
