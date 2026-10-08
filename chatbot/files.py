import io

from pypdf import PdfReader

MAX_DOC_CHARS = 12000


def extract_text(filename, data):
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:30])
    else:
        text = data.decode("utf-8", errors="replace")
    return text[:MAX_DOC_CHARS]
