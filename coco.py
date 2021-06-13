def load_classes(names_path):
    with open(names_path, 'r') as names_file:
        names = [line.strip() for line in names_file]
    return names
