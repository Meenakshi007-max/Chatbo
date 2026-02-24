
from langdetect import detect
import re


def split_sentences(text):
    """
    Sentence splitter for English + Indic languages
    Handles . ! ? and Hindi danda (।)
    """
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk_pages(pages, chunk_size=500, overlap=100, source=""):
    chunks = []
    chunk_id = 1

    for page in pages:
        page_number = page["page"]

        # Paragraph split
        paragraphs = re.split(r'\n\s*\n', page["text"])

        for para in paragraphs:
            if not para.strip():
                continue

            # Language detection
            try:
                lang = detect(para)
            except:
                lang = "unknown"

            sentences = split_sentences(para)

            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip(),
                        "page": page_number,
                        "source": source,
                        "language": lang
                    })
                    chunk_id += 1

                    # overlap
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence

            if current_chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "page": page_number,
                    "source": source,
                    "language": lang
                })
                chunk_id += 1

    return chunks