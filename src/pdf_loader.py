from pathlib import Path
from pypdf import PdfReader


def loading_documents(folder_path):
    """
    Load all PDF documents from the knowledge base.

    Each page is stored separately so that page-level
    citations can be generated later.
    """
    # list to store documents retreived
    docs = []
    # path of the folder
    folder = Path(folder_path)

    for file in folder.iterdir():
        # check for pdfs
        if file.suffix.lower() == ".pdf":
            # read the pdf
            reader = PdfReader(file)
        # for every page in the document
            for page_number, page in enumerate(reader.pages, start=1):
                # extract the text
                page_text = page.extract_text()
                # store the extracted text, file name and page number
                if page_text:
                    docs.append(
                        {
                            "filename": file.name,
                            "page": page_number,
                            "content": page_text.strip()
                        }
                    )

    return docs

