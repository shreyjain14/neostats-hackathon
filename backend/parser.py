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
            return str(file.read(), "utf-8")
        else:
            raise ValueError("Unsupported file format")
    
    def split_into_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Split contract text into individual clauses"""
        # Split by common clause separators
        clause_separators = [
            r'\n\s*\d+\.',  # Numbered clauses
            r'\n\s*[A-Z]\.',  # Lettered clauses
            r'\n\s*\([a-z]\)',  # Sub-clauses
            r'\n\s*Article\s+\d+',  # Articles
            r'\n\s*Section\s+\d+'  # Sections
        ]
        
        # Combine all separators
        pattern = '|'.join(clause_separators)
        clauses_raw = re.split(pattern, text, flags=re.IGNORECASE)
        
        clauses = []
        for i, clause_text in enumerate(clauses_raw):
            clause_text = clause_text.strip()
            if len(clause_text) > 50:  # Filter out very short clauses
                clause_type = self.classify_clause(clause_text)
                clauses.append({
                    "id": i,
                    "text": clause_text,
                    "type": clause_type,
                    "word_count": len(clause_text.split())
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
    
    def extract_key_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract key terms and entities from contract"""
        key_terms = {
            "dates": [],
            "amounts": [],
            "parties": [],
            "locations": []
        }
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b\w+ \d{1,2}, \d{4}\b'
        ]
        for pattern in date_patterns:
            key_terms["dates"].extend(re.findall(pattern, text))
        
        # Extract monetary amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        key_terms["amounts"] = re.findall(amount_pattern, text)
        
        # Extract potential party names (capitalized phrases)
        party_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*(?:,? (?:Inc|LLC|Corp|Ltd|Company)\.?)?'
        key_terms["parties"] = list(set(re.findall(party_pattern, text)))
        
        return key_terms

contract_parser = ContractParser()
