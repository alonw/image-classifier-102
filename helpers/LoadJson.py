# LABEL MAPPING
import json

def load_cat_to_names_json(cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name