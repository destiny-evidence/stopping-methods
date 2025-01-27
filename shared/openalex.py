def revert_index(inverted_index: dict[str, list[int]] | None) -> str | None:
    if inverted_index is None:
        return None

    token: str
    position: int
    positions: list[int]

    abstract_length: int = len([1 for idxs in inverted_index.values() for _ in idxs])
    abstract: list[str] = [''] * abstract_length

    for token, positions in inverted_index.items():
        for position in positions:
            if position < abstract_length:
                abstract[position] = token

    return ' '.join(abstract)
