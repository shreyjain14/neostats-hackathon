from typing import List, Dict, Any, Tuple
import numpy as np
from openai import AzureOpenAI
from config.settings import settings
from backend.embeddings import embedding_manager
import json
import logging

class RiskScorer:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.business_embeddings = embedding_manager.embed_business_requirements()
    
    def analyze_clause_risk(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk for a single clause"""
        try:
            # Get AI-powered risk analysis
            risk_analysis = self._get_ai_risk_analysis(clause['text'], clause['type'])
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(clause)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(risk_analysis, compliance_score)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "compliance_score": compliance_score,
                "compliance_status": self._get_compliance_status(compliance_score),
                "analysis": risk_analysis,
                "flags": self._identify_risk_flags(clause['text'])
            }
        except Exception as e:
            logging.error(f"Error analyzing clause risk: {e}")
            return self._default_risk_analysis()
    
    def _get_ai_risk_analysis(self, clause_text: str, clause_type: str) -> Dict[str, Any]:
        """Get AI-powered risk analysis for a clause"""
        try:
            prompt = f"""
            Analyze the following contract clause for potential risks and issues from the perspective of "Hari and Winston Associates LLC". Our company provides data analytics, ML deployment, and dashboard development services. The contract should protect our interests.

            Clause Type: {clause_type}
            Clause Text: {clause_text}

            Evaluate the clause based on how it impacts Hari and Winston Associates LLC. Is it neutral, in our favor, or does it pose a risk to us?

            Consider these business requirements for Hari and Winston Associates LLC:
            - Services: Data analytics, ML deployment, dashboard development
            - Business model: Fixed-fee contracts with milestone payments
            - Key Protections: Limited liability, clear IP ownership, and defined scope to prevent scope creep.

            Respond in JSON format with:
            {{
                "completeness_score": 0.0-1.0,
                "compliance_score": 0.0-1.0,
                "risk_exposure": 0.0-1.0,  // Higher score means more risk to Hari and Winston Associates LLC
                "clarity_score": 0.0-1.0,
                "cost_benefit_score": 0.0-1.0,
                "issues": ["list of identified issues posing a risk to us"],
                "concerns": ["list of concerns for our company"],
                "recommendations": ["list of recommendations to protect our interests"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a legal contract analyst working for Hari and Winston Associates LLC. Your goal is to identify risks to the company."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error getting AI risk analysis: {e}")
            return self._default_ai_analysis()
    
    def _calculate_compliance_score(self, clause: Dict[str, Any]) -> float:
        """Calculate compliance score based on business requirements"""
        try:
            clause_embedding = embedding_manager.generate_embedding(clause['text'])
            
            # Check alignment with business requirements
            max_similarity = 0.0
            clause_type = clause['type']
            
            if clause_type in ['scope_of_work', 'general']:
                # Check against services and deliverables
                for category in ['services', 'deliverables']:
                    if category in self.business_embeddings:
                        for req_embedding in self.business_embeddings[category]:
                            similarity = embedding_manager.cosine_similarity(
                                clause_embedding, req_embedding
                            )
                            max_similarity = max(max_similarity, similarity)
            
            return min(max_similarity, 1.0)
        except Exception as e:
            logging.error(f"Error calculating compliance score: {e}")
            return 0.5
    
    def _calculate_risk_score(self, ai_analysis: Dict[str, Any], compliance_score: float) -> float:
        """Calculate overall risk score"""
        try:
            # Weight different factors
            weights = {
                'completeness': 0.2,
                'compliance': 0.2,
                'risk_exposure': 0.3,
                'clarity': 0.2,
                'cost_benefit': 0.1
            }
            
            # Calculate weighted score (lower is better for risk)
            risk_components = [
                (1 - ai_analysis.get('completeness_score', 0.5)) * weights['completeness'],
                (1 - ai_analysis.get('compliance_score', 0.5)) * weights['compliance'],
                ai_analysis.get('risk_exposure', 0.5) * weights['risk_exposure'],
                (1 - ai_analysis.get('clarity_score', 0.5)) * weights['clarity'],
                (1 - ai_analysis.get('cost_benefit_score', 0.5)) * weights['cost_benefit']
            ]
            
            base_risk = sum(risk_components)
            
            # Adjust for compliance
            compliance_adjustment = (1 - compliance_score) * 0.2
            
            total_risk = min(base_risk + compliance_adjustment, 1.0)
            return total_risk
        except Exception as e:
            logging.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score >= settings.RISK_THRESHOLD_HIGH:
            return "HIGH"
        elif risk_score >= settings.RISK_THRESHOLD_MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_compliance_status(self, compliance_score: float) -> str:
        """Determine compliance status"""
        if compliance_score >= 0.8:
            return "COMPLIANT"
        elif compliance_score >= 0.6:
            return "PARTIAL"
        else:
            return "NON_COMPLIANT"
    
    def _identify_risk_flags(self, clause_text: str) -> List[str]:
        """Identify specific risk flags in clause text"""
        flags = []
        text_lower = clause_text.lower()
        
        risk_patterns = {
            "unlimited_liability": ["unlimited liability", "no limitation of liability"],
            "broad_indemnification": ["indemnify", "hold harmless", "defend"],
            "vague_termination": ["terminate at will", "terminate for convenience"],
            "one_sided_ip": ["all rights", "exclusive ownership"],
            "penalty_clauses": ["penalty", "liquidated damages"],
            "automatic_renewal": ["automatic renewal", "auto-renew"],
            "broad_confidentiality": ["all information", "any information"]
        }
        
        for flag_type, patterns in risk_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    flags.append(flag_type)
                    break
        
        return flags
    
    def calculate_overall_contract_score(self, clause_analyses: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Calculate overall contract risk score"""
        if not clause_analyses:
            return 0.5, "MEDIUM"
        
        # Weight clauses by importance
        clause_weights = {
            "liability": 0.25,
            "termination": 0.20,
            "intellectual_property": 0.15,
            "payment": 0.15,
            "confidentiality": 0.10,
            "scope_of_work": 0.10,
            "general": 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for analysis in clause_analyses:
            clause_type = analysis.get('clause_type', 'general')
            weight = clause_weights.get(clause_type, 0.05)
            weighted_score += analysis['risk_score'] * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = np.mean([a['risk_score'] for a in clause_analyses])
        
        risk_level = self._determine_risk_level(overall_score)
        
        return overall_score, risk_level
    
    def _default_risk_analysis(self) -> Dict[str, Any]:
        """Default risk analysis when AI analysis fails"""
        return {
            "risk_score": 0.5,
            "risk_level": "MEDIUM",
            "compliance_score": 0.5,
            "compliance_status": "PARTIAL",
            "analysis": self._default_ai_analysis(),
            "flags": []
        }
    
    def _default_ai_analysis(self) -> Dict[str, Any]:
        """Default AI analysis structure"""
        return {
            "completeness_score": 0.5,
            "compliance_score": 0.5,
            "risk_exposure": 0.5,
            "clarity_score": 0.5,
            "cost_benefit_score": 0.5,
            "issues": ["Unable to analyze - using default assessment"],
            "concerns": ["Manual review recommended"],
            "recommendations": ["Have legal team review this clause"]
        }

risk_scorer = RiskScorer()