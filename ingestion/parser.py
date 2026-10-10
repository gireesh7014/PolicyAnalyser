import fitz  # PyMuPDF
from typing import List, Dict, Any
import base64
import io

class PDFParser:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_base64_pdf(self, base64_pdf: str) -> List[Dict[str, Any]]:
        """
        Parses a base64 encoded PDF and returns a list of text chunks with their bounding boxes.
        """
        pdf_data = base64.b64decode(base64_pdf)
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        chunks = []
        current_chunk_text = ""
        current_chunk_bboxes = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # We only want text blocks (type 0)
                if block[6] != 0:
                    continue
                    
                bbox = {
                    "page": page_num + 1,
                    "x0": block[0],
                    "y0": block[1],
                    "x1": block[2],
                    "y1": block[3]
                }
                text = block[4].strip()
                
                if not text:
                    continue

                if len(current_chunk_text) + len(text) > self.chunk_size:
                    # Save current chunk
                    if current_chunk_text:
                        chunks.append({
                            "text": current_chunk_text,
                            "bboxes": current_chunk_bboxes
                        })
                    
                    # Start new chunk with overlap logic (simplified by just starting fresh here for now, 
                    # advanced overlap could split sentences)
                    current_chunk_text = text + " "
                    current_chunk_bboxes = [bbox]
                else:
                    current_chunk_text += text + " "
                    current_chunk_bboxes.append(bbox)
                    
        # Add the last chunk
        if current_chunk_text.strip():
            chunks.append({
                "text": current_chunk_text.strip(),
                "bboxes": current_chunk_bboxes
            })
            
        return chunks
