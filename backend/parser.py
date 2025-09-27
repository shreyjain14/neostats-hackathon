import PyPDF2
import docx
from typing import List, Dict, Any
import re
import logging

class ContractParser:
    def __init__(self):
        self.clause_patterns = {
            "termination": [
                r"termination", r"terminate", r"end of agreement", r"expiry"
            ],
            "liability": [
                r"liability", r"liable", r"damages", r"indemnity", r"indemnification"
            ],
            "intellectual_property": [
                r"intellectual property", r"IP", r"copyright", r"patent", r"trademark"
            ],
            "payment": [
                r"payment", r"fee", r"compensation", r"invoice", r"billing"
            ],
            "confidentiality": [
                r"confidential", r"non-disclosure", r"proprietary", r"secret"
            ],
            "scope_of_work": [
                r"scope", r"services", r"deliverables", r"work product"
            ],
            "force_majeure": [
                r"force majeure", r"acts of god", r"unforeseeable circumstances"
            ],
            "governing_law": [
                r"governing law", r"jurisdiction", r"legal disputes"
            ]
        }
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting PDF text: {e}")
            raise
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting DOCX text: {e}")
            raise
    
    def extract_text(self, file, filename: str) -> str:
        """Extract text based on file type"""
        if filename.endswith('.pdf'):
            return self.extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            return self.extract_text_from_docx(file)
        elif filename.endswith('.txt'):
            return file.read().decode("utf-8")
        else:
            raise ValueError("Unsupported file format")
    
    def split_into_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Split contract text into individual clauses with improved logic."""
        clauses = []
        
        # Primary method: Split by common clause headings (e.g., "1. ...", "A. ...", "Article 1 ...")
        # This regex looks for a line break, optional whitespace, a number/letter, a period, and then a space.
        pattern = r'\n\s*(\d+\.|\([a-zA-Z]\)|[A-Z]\.)\s+'
        clauses_raw = re.split(pattern, text)
        
        # Post-process the split to combine the delimiter with the clause text
        if len(clauses_raw) > 1:
            processed_clauses = []
            # Skip the first element if it's empty (often happens with re.split)
            it = iter(clauses_raw[1:])
            for delimiter in it:
                try:
                    content = next(it)
                    processed_clauses.append(delimiter.strip() + " " + content.strip())
                except StopIteration:
                    break
            clauses_raw = processed_clauses

        # Fallback method for plain text: Split by double newlines (paragraphs)
        if len(clauses_raw) <= 1:
            clauses_raw = text.split('\n\n')

        # Final processing and classification
        for i, clause_text in enumerate(clauses_raw):
            clause_text = clause_text.strip()
            # Filter out very short, likely irrelevant parts
            if len(clause_text.split()) > 10: 
                clause_type = self.classify_clause(clause_text)
                clauses.append({
                    "id": i,
                    "text": clause_text,
                    "type": clause_type,
                    "word_count": len(clause_text.split())
                })
        
        # If still no clauses found, treat the whole document as one 'general' clause.
        if not clauses:
            clauses.append({
                "id": 0,
                "text": text.strip(),
                "type": "general",
                "word_count": len(text.strip().split())
            })
            
        return clauses
    
    def classify_clause(self, clause_text: str) -> str:
        """Classify clause type based on content"""
        clause_text_lower = clause_text.lower()
        
        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, clause_text_lower):
                    return clause_type
        
        return "general"

contract_parser = ContractParser()