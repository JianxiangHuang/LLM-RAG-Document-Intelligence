from functools import lru_cache

from tiktoken import encoding_for_model, get_encoding


@lru_cache(maxsize=4)
def _get_token_encoding(model: str):
    try:
        encoding = encoding_for_model(model)
    except KeyError:
        encoding = get_encoding("cl100k_base")
    return encoding


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if not isinstance(model, str):
        raise TypeError("Model must be a string")
    if text == "":
        return 0

    return len(_get_token_encoding(model).encode(text))


