from services.embedding_service import embed_texts
import math


def score_semantic_boundaries(atomic_units: list[dict]) -> list[dict]:
    windows = _build_boundary_windows(atomic_units)
    windows_with_embeddings = _get_windows_embeddings(windows)
    return _calculate_semantic_boundary_scores(windows_with_embeddings)


def _build_boundary_windows(atomic_units: list[dict]) -> list[dict]:
    _validate_atomic_units(atomic_units)

    windows = []
    unit_size = len(atomic_units)
    for index in range(unit_size - 1):
        left_indexes = range(max(0, index - 2), index + 1)
        right_indexes = range(index + 1, min(index + 4, unit_size))
        windows.append({
            "left_window_text": "\n".join([atomic_units[i]["text"] for i in left_indexes]),
            "right_window_text": "\n".join([atomic_units[i]["text"] for i in right_indexes]),
            "unit_index": index,
            "left_unit_indexes": list(left_indexes),
            "right_unit_indexes": list(right_indexes),
        })
    return windows


def _get_windows_embeddings(windows: list[dict]) -> list[dict]:
    if not windows:
        return []
    left_window_texts = [window["left_window_text"] for window in windows]
    right_window_texts = [window["right_window_text"] for window in windows]
    left_window_embeddings = embed_texts(left_window_texts)
    right_window_embeddings = embed_texts(right_window_texts)
    return [
        {
            **window,
            "left_embedding": left_embedding,
            "right_embedding": right_embedding,
        }
        for window, left_embedding, right_embedding in zip(
            windows,
            left_window_embeddings,
            right_window_embeddings,
        )
    ]


def _calculate_semantic_boundary_scores(windows_with_embeddings: list[dict]) -> list[dict]:
    scored_windows = []
    for window in windows_with_embeddings:
        left_embedding = window["left_embedding"]
        right_embedding = window["right_embedding"]
        similarity = _cosine_similarity(left_embedding, right_embedding)
        distance = 1 - similarity
        scored_windows.append({
            "window_index": window["unit_index"],
            "distance": distance,
            "similarity": similarity,
            "left_unit_indexes": window["left_unit_indexes"],
            "right_unit_indexes": window["right_unit_indexes"],
            "left_window_text": window["left_window_text"],
            "right_window_text": window["right_window_text"],
        })
    return scored_windows


def _cosine_similarity(left_embedding: list[float],right_embedding: list[float],) -> float:

    if not left_embedding:
        raise ValueError("Embeddings cannot be empty.")
    if not right_embedding:
        raise ValueError("Embeddings cannot be empty.")

    if len(left_embedding) != len(right_embedding):
        raise ValueError("Embedding dimensions must match.")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_embedding,
            right_embedding,
        )
    )

    left_norm = math.sqrt(
        sum(value * value for value in left_embedding)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right_embedding)
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding norm cannot be zero.")

    similarity = dot_product / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))




def _validate_atomic_units(atomic_units: list[dict]) -> None:
    if not isinstance(atomic_units, list):
        raise TypeError("Atomic units must be a list.")

    for index, unit in enumerate(atomic_units):
        if not isinstance(unit, dict):
            raise TypeError(f"Atomic unit at index {index} must be a dict.")

        text = unit.get("text")
        if not isinstance(text, str):
            raise TypeError(
                f"Atomic unit text at index {index} must be a string."
            )

        if not text.strip():
            raise ValueError(
                f"Atomic unit text at index {index} cannot be empty."
            )