from rag.pdf_processor import extract_text_from_pdf


pdf_path = "data/uploads/dbms_notes.pdf"

pages = extract_text_from_pdf(pdf_path)


print("Total pages extracted:", len(pages))


for page in pages[:3]:

    print("\n" + "=" * 60)

    print("PAGE:", page["page_number"])

    print("=" * 60)

    print(page["text"][:1000])