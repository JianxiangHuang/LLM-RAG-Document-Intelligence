import math


def simple_chunking(text: str, chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    if not text:
        raise ValueError("Text is empty")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap < 0:
        raise ValueError("Overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    step = chunk_size - overlap

    for chunk_index, start_char in enumerate(range(0, len(text), step)):
        end_char = min(start_char + chunk_size, len(text))
        chunk_content = text[start_char:end_char]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_content,
                "char_count": len(chunk_content),
                "source_position": {
                    "type": "text_offset",
                    "start_char": start_char,
                    "end_char": end_char,
                },
            }
        )

    return chunks


BIG_MAX_TOKENS = 6000
SMALL_MIN_TOKENS = 50
SMALL_MAX_TOKENS = 600


def big_small_chunking(original_text: str, atomic_units: list[dict], boundary_scores: list[dict], ):
    big_chunks = _build_big_chunks(original_text, atomic_units, boundary_scores, )
    all_small_chunks=[]
    for big_chunk in big_chunks:
        current_small_chunks=_build_small_chunks(original_text, atomic_units, boundary_scores, big_chunk["start_unit_index"], big_chunk["end_unit_index"], big_chunk["chunk_index"])
        if current_small_chunks:
            all_small_chunks.extend(current_small_chunks)
    return big_chunks,all_small_chunks



def _build_big_chunks(original_text: str, atomic_units: list[dict], boundary_scores: list[dict], ) -> list[dict]:
    if not atomic_units:
        return []

    if len(atomic_units) == 1:
        unit = atomic_units[0]
        return [_build_big_chunk(original_text[unit["start_char"]:unit["end_char"]],
                                 0, {"type": "text_offset",
                                            "start_char": unit["start_char"],
                                            "end_char": unit["end_char"], },0,0)]

    if len(boundary_scores) != len(atomic_units) - 1:
        raise ValueError("Boundary score count must equal atomic unit count minus one.")

    original_distances = [boundary_score["distance"] for boundary_score in boundary_scores]
    processed_distances = _normalize_and_sort_distances(original_distances)

    big_chunks = []
    start_unit_index = 0
    current_token_count = 0
    chunk_index = 0
    max_tokens = 6000

    def emit_current_chunk(end_unit_index: int) -> None:
        nonlocal chunk_index

        start_char = atomic_units[start_unit_index]["start_char"]
        end_char = atomic_units[end_unit_index]["end_char"]
        chunk_text = original_text[start_char:end_char]

        big_chunks.append(_build_big_chunk(chunk_text, chunk_index,
                                           {"type": "text_offset",
                                            "start_char": start_char,
                                            "end_char": end_char, },
                                           start_unit_index, end_unit_index))
        chunk_index += 1

    for index, atomic_unit in enumerate(atomic_units):
        current_token_count += atomic_unit["token_count"]

        if index == len(atomic_units) - 1:
            emit_current_chunk(index)
            break

        next_token_count = atomic_units[index + 1]["token_count"]

        must_force_cut = (current_token_count + next_token_count > max_tokens)

        should_semantic_cut = False

        if not must_force_cut:
            percentile = _dynamic_percentile_big(current_token_count)
            threshold = _calculate_percentile(processed_distances, percentile, )

            boundary_distance = boundary_scores[index]["distance"]

            should_semantic_cut = boundary_distance > threshold

        if must_force_cut or should_semantic_cut:
            emit_current_chunk(index)
            start_unit_index = index + 1
            current_token_count = 0

    return big_chunks


def _build_big_chunk(chunk_text: str, chunk_index, source_position: dict, start_unit_index: int,
                     end_unit_index: int) -> dict:
    return {
        "chunk_index": chunk_index,
        "chunk_method": "semantic",
        "chunk_type": "big",
        "text": chunk_text,
        "char_count": len(chunk_text),
        "source_position": source_position,
        "start_unit_index": start_unit_index,
        "end_unit_index": end_unit_index,
    }


def _build_small_chunks(original_text: str, atomic_units: list[dict], boundary_scores: list[dict], left_index: int,
                        right_index: int, parent_id: int) -> list[dict] | None:
    if left_index < 0 or right_index < 0:
        raise ValueError("start_unit_index and end_unit_index must be greater than or equal to 0.")
    if left_index > right_index:
        raise ValueError("start_unit_index must be less than or equal to end_unit_index.")
    if right_index >= len(atomic_units):
        raise ValueError("end_unit_index is outside atomic units.")

    if right_index - left_index == 0:
        return None

    if right_index - left_index == 1 and atomic_units[left_index]["token_count"]>=SMALL_MIN_TOKENS and atomic_units[right_index]["token_count"]>=SMALL_MIN_TOKENS:
        start_char_left = atomic_units[left_index]["start_char"]
        end_char_left = atomic_units[left_index]["end_char"]
        start_char_right = atomic_units[right_index]["start_char"]
        end_char_right = atomic_units[right_index]["end_char"]
        return [
            _build_small_chunk(original_text[start_char_left:end_char_left],0,
                               {"type": "text_offset",
                                "start_char": start_char_left,
                                "end_char": end_char_left, },parent_id
                               ),
            _build_small_chunk(original_text[start_char_right:end_char_right],1,
                               {"type": "text_offset",
                                "start_char": start_char_right,
                                "end_char": end_char_right, },parent_id)
        ]

    internal_scores = [score for score in boundary_scores if left_index <= score["window_index"] < right_index]

    expected_window_indexes = set(range(left_index, right_index))
    actual_window_indexes = [score["window_index"] for score in internal_scores]

    if (len(actual_window_indexes) != len(expected_window_indexes)
            or set(actual_window_indexes) != expected_window_indexes):
        missing_indexes = sorted(expected_window_indexes - set(actual_window_indexes))

        raise ValueError(
            "Internal boundary scores are incomplete or duplicated. "
            f"Missing window indexes: {missing_indexes}")

    processed_distances = _normalize_and_sort_distances([score["distance"] for score in internal_scores])

    score_by_window_index = {score["window_index"]: score["distance"]for score in internal_scores}

    small_chunks = []
    start_unit_index = 0
    current_token_count = 0
    chunk_index = 0
    max_tokens = 600

    def emit_current_chunk(end_unit_index: int) -> None:
        nonlocal chunk_index

        start_char = big_chunk_units[start_unit_index]["start_char"]
        end_char = big_chunk_units[end_unit_index]["end_char"]
        chunk_text = original_text[start_char:end_char]

        small_chunks.append(_build_small_chunk(chunk_text, chunk_index,
                                               {"type": "text_offset",
                                                "start_char": start_char,
                                                "end_char": end_char, },
                                               parent_id))
        chunk_index += 1

    big_chunk_units = atomic_units[left_index:right_index + 1]
    for index, atomic_unit in enumerate(big_chunk_units):
        current_token_count += atomic_unit["token_count"]

        if index == len(big_chunk_units) - 1:
            emit_current_chunk(index)
            break

        next_token_count = big_chunk_units[index + 1]["token_count"]

        must_force_cut = (current_token_count + next_token_count > max_tokens)

        should_semantic_cut = False

        if not must_force_cut:
            percentile = _dynamic_percentile_small(current_token_count)
            threshold = _calculate_percentile(processed_distances, percentile, )


            global_boundary_index = left_index + index
            boundary_distance = score_by_window_index[global_boundary_index]

            should_semantic_cut = boundary_distance > threshold

        if must_force_cut or should_semantic_cut:
            emit_current_chunk(index)
            start_unit_index = index + 1
            current_token_count = 0

    return small_chunks


def _build_small_chunk(chunk_text: str, chunk_index, source_position: dict, parent_id: int) -> dict:
    return {
        "chunk_index": chunk_index,
        "chunk_method": "semantic",
        "chunk_type": "small",
        "text": chunk_text,
        "char_count": len(chunk_text),
        "source_position": source_position,
        "parent_chunk_index": parent_id,
    }


def _dynamic_percentile_small(tokens: int) -> float:
    min_token = SMALL_MIN_TOKENS
    mid_token = 200
    max_token = SMALL_MAX_TOKENS
    percentile_at_min = 0.7
    percentile_at_mid = 0.5
    percentile_at_limit = 0.05
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("Tokens must be an integer greater than 0.")
    if tokens < min_token:
        return 1
    if tokens < mid_token:
        return percentile_at_min - (tokens - min_token) / (mid_token - min_token) * (
                    percentile_at_min - percentile_at_mid)
    if tokens < 2 * mid_token:
        return percentile_at_mid - ((tokens - mid_token) / mid_token) ** 2 * (percentile_at_mid - percentile_at_limit)
    if tokens < max_token:
        return percentile_at_limit
    return 0


def _dynamic_percentile_big(tokens: int) -> float:
    min_token = 500
    mid_token = 2000
    max_token = BIG_MAX_TOKENS
    percentile_at_min = 0.9
    percentile_at_mid = 0.75
    percentile_at_limit = 0.05
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("Tokens must be an integer greater than 0.")
    if tokens < min_token:
        return 1
    if tokens < mid_token:
        return percentile_at_min - (tokens - min_token) / (mid_token - min_token) * (
                    percentile_at_min - percentile_at_mid)
    if tokens < 2 * mid_token:
        return percentile_at_mid - ((tokens - mid_token) / mid_token) ** 2 * (percentile_at_mid - percentile_at_limit)
    if tokens < max_token:
        return percentile_at_limit
    return 0


def _calculate_percentile(distances: list[float], percentile: float, ) -> float:
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise TypeError("Percentile must be a number.")

    if not 0 <= percentile <= 1:
        raise ValueError("Percentile must be between 0 and 1.")

    if not distances:
        raise ValueError("Distances cannot be empty.")

    if len(distances) == 1:
        return distances[0]

    position = (len(distances) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return distances[lower_index]

    weight = position - lower_index

    return (
            distances[lower_index] * (1 - weight)
            + distances[upper_index] * weight
    )


def _normalize_and_sort_distances(distances: list[float]):
    if not isinstance(distances, list):
        raise TypeError("Distances must be a list.")

    if not distances:
        raise ValueError("Distances cannot be empty.")

    normalized_distances = []
    for distance in distances:
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise TypeError("Each distance must be a number.")

        if not 0 <= distance <= 2:
            raise ValueError("Each distance must be between 0 and 2.")

        normalized_distances.append(distance)

    return sorted(normalized_distances)
