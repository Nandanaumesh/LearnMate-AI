def create_chunks(pages, chunk_size=500, overlap=100):
    """
    Split PDF text into overlapping chunks.

    Args:
        pages: List of dictionaries containing page number and text.
        chunk_size: Number of words in each chunk.
        overlap: Number of overlapping words.

    Returns:
        List of chunk dictionaries.
    """

    chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "page_number": page["page_number"]
            })

            start += chunk_size - overlap

    return chunks