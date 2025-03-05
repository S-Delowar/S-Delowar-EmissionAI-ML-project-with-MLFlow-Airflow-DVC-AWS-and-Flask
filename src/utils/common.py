import yaml

def read_yml(filepath):
    with open(filepath, 'r') as file:
        content_dict = yaml.safe_load(file)
    return content_dict