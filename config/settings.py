import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "contract_risk_db")
    DB_USER = os.getenv("DB_USER", "myuser")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
    
    # Application
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", 0.7))
    RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", 0.4))
    
    # Business Requirements Schema
    BUSINESS_REQUIREMENTS = {
        "services": [
            "Data analytics consulting and strategy",
            "Dashboard development and visualization (e.g., Power BI, Tableau)",
            "Machine learning and predictive model deployment",
            "Data pipeline design, optimization, and integration",
            "Ongoing technical support, system maintenance, and model updates"
        ],
        "deliverables": [
            "Custom dashboards",
            "Technical documentation",
            "Monthly performance reports",
            "ML model updates"
        ],
        "business_considerations": [
            "Confidentiality of client data",
            "Right to use anonymized insights",
            "Intellectual property retention",
            "Fixed-fee contracts with milestone payments",
            "Limited liability clauses"
        ],
        "risk_areas": [
            "Liability limitations and exclusions",
            "Termination clauses with notice requirements",
            "Intellectual property ownership",
            "Payment terms and late fees",
            "Confidentiality obligations",
            "Project scope and timeline adjustments"
        ]
    }

settings = Settings()
