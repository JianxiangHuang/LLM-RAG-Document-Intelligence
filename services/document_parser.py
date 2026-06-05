from pathlib import Path


def parse_text(path: Path)-> str:
    return path.read_text(encoding="utf-8")

def parse_markdown(path: Path)-> str:
    return path.read_text(encoding="utf-8")



SUPPORTED_DOCUMENT_EXTENSIONS = {'.txt':parse_text,
                                 '.markdown':parse_markdown,
                                 '.md':parse_markdown,}




def parse_document(document_path: Path)->str:
    path=Path(document_path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    if not path.is_file():
        raise ValueError(f"{path} is not a file.")

    file_extension = path.suffix.lower()
    parser=SUPPORTED_DOCUMENT_EXTENSIONS.get(file_extension)
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return parser(path)


