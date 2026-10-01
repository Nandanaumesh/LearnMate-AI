import pymupdf

def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        list of dictionaries containing:
        - page number
        - page text
    """

    document =pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "page_number": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages