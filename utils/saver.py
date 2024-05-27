import pathlib

def prepare_path(path: str):
    split_path = path.split("/")
    filename = split_path[-1]
    folder_path = "/".join(split_path[:-1])

    base = pathlib.Path(folder_path)
    base.mkdir(parents=True, exist_ok=True)

    return base / filename
