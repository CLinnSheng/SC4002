import re

import numpy as np


def get_context_average_embedding(
    sentence_tokens: List[str],
    oov_token: str,
    glove: Any = None,
    embedding_matrix: np.ndarray = None,
    index_from_word: dict = None,
) -> np.ndarray:
    """Generates an approximate embedding for an OOV word by averaging the
    embeddings of surrounding context words.

    Args:
        sentence_tokens (List[str]): A list of tokens from a sentence.
        oov_token (str): The OOV word for which an approximate embedding is needed.
        glove : A pretrained glove embedding model
        embedding_matrix (np.ndarray, optional): Precomputed embedding matrix.
        index_from_word (dict, optional): Dictionary mapping words to row indices
            in the embedding_matrix.

    Returns:
        np.ndarray: The averaged embedding vector for the OOV word.
        Returns a zero vector if no in-vocabulary context words are found.
    """

    def _get_embedding(word: str) -> np.ndarray:
        """Internal helper to get embedding from any model and convert to numpy."""
        if word and word in glove.stoi:
            return glove[word]
        elif (
            embedding_matrix is not None
            and index_from_word is not None
            and word in index_from_word
        ):
            return embedding_matrix[index_from_word[word]]

        return None

    context_embeddings = [
        _get_embedding(word)
        for word in sentence_tokens
        if word != oov_token and _get_embedding(word) is not None
    ]

    if context_embeddings:
        return np.mean(context_embeddings, axis=0)

    return np.zeros(glove.dim)
