from typing import List, Dict, Any
from openai import AzureOpenAI
from config.settings import settings
import json
import logging

class SuggestionGenerator:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
    
    def generate_clause_suggestions(self, clause_analysis: Dict[str, Any], clause_text: str, clause_type: str) -> List[Dict[str, Any]]:
        """Generate suggestions for improving a specific clause"""
        try:
            prompt = f"""
            Based on the risk analysis of this contract clause, provide specific suggestions for improvement:
            
            Clause Type: {clause_type}
            Risk Level: {clause_analysis['risk_level']}
            Risk Score: {clause_analysis['risk_score']:.2f}
            Compliance Status: {clause_analysis['compliance_status']}
            
            Current Clause Text:
            {clause_text}
            
            Risk Flags: {', '.join(clause_analysis.get('flags', []))}
            
            Business Context:
            - We provide data analytics, ML deployment, and dashboard development services
            - We work on fixed-fee contracts with milestone payments
            - We need to protect our IP while allowing client data usage
            - We require clear scope definition to avoid scope creep
            
            Provide suggestions in JSON format:
            {{
                "suggestions": [
                    {{
                        "priority": "HIGH|MEDIUM|LOW",
                        "category": "liability|payment|scope|ip|termination|other",
                        "issue": "description of the issue",
                        "suggestion": "specific suggestion for improvement",
                        "alternative_wording": "suggested alternative clause text (if applicable)",
                        "rationale": "why this change would reduce risk"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a legal advisor specializing in technology services contracts. Provide practical, actionable suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            suggestions_data = json.loads(response.choices[0].message.content)
            return suggestions_data.get('suggestions', [])
        except Exception as e:
            logging.error(f"Error generating clause suggestions: {e}")
            return self._get_default_suggestions(clause_analysis['risk_level'])
    
    def generate_contract_level_suggestions(self, contract_analysis: Dict[str, Any], clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate contract-level improvement suggestions"""
        try:
            # Analyze overall contract structure
            high_risk_clauses = [c for c in clauses if c.get('risk_level') == 'HIGH']
            missing_clauses = self._identify_missing_clauses(clauses)
            
            prompt = f"""
            Provide contract-level suggestions for this technology services agreement:
            
            Overall Risk Score: {contract_analysis.get('overall_risk_score', 0):.2f}
            Overall Risk Level: {contract_analysis.get('risk_level', 'UNKNOWN')}
            
            High Risk Clauses: {len(high_risk_clauses)}
            Potentially Missing Clauses: {', '.join(missing_clauses)}
            
            Business Requirements:
            - Data analytics and ML services
            - Dashboard development (Power BI, Tableau)
            - Fixed-fee milestone-based contracts
            - IP protection while allowing anonymized data use
            - Clear deliverables and timelines
            
            Provide contract-level suggestions in JSON format:
            {{
                "structural_suggestions": [
                    {{
                        "priority": "HIGH|MEDIUM|LOW",
                        "category": "structure|missing_clauses|risk_mitigation|compliance",
                        "issue": "description of the structural issue",
                        "suggestion": "specific recommendation",
                        "impact": "how this would improve the contract"
                    }}
                ],
                "missing_clauses": [
                    {{
                        "clause_type": "type of missing clause",
                        "importance": "HIGH|MEDIUM|LOW",
                        "suggested_content": "what should be included",
                        "rationale": "why this clause is needed"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a contract specialist for technology services companies. Focus on practical business protection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            suggestions_data = json.loads(response.choices[0].message.content)
            
            # Combine structural and missing clause suggestions
            all_suggestions = suggestions_data.get('structural_suggestions', [])
            for missing in suggestions_data.get('missing_clauses', []):
                all_suggestions.append({
                    "priority": missing['importance'],
                    "category": "missing_clause",
                    "issue": f"Missing {missing['clause_type']} clause",
                    "suggestion": missing['suggested_content'],
                    "alternative_wording": "",
                    "rationale": missing['rationale']
                })
            
            return all_suggestions
        except Exception as e:
            logging.error(f"Error generating contract suggestions: {e}")
            return self._get_default_contract_suggestions()
    
    def _identify_missing_clauses(self, clauses: List[Dict[str, Any]]) -> List[str]:
        """Identify potentially missing clause types"""
        present_types = set(clause.get('type', 'general') for clause in clauses)
        
        essential_clauses = {
            'liability', 'termination', 'intellectual_property', 
            'payment', 'confidentiality', 'scope_of_work',
            'force_majeure', 'governing_law'
        }
        
        missing = essential_clauses - present_types
        return list(missing)
    
    def _get_default_suggestions(self, risk_level: str) -> List[Dict[str, Any]]:
        """Default suggestions when AI generation fails"""
        if risk_level == "HIGH":
            return [
                {
                    "priority": "HIGH",
                    "category": "other",
                    "issue": "High risk clause detected",
                    "suggestion": "Review this clause with legal counsel",
                    "alternative_wording": "",
                    "rationale": "High risk clauses require professional review"
                }
            ]
        else:
            return [
                {
                    "priority": "MEDIUM",
                    "category": "other",
                    "issue": "Standard review recommended",
                    "suggestion": "Consider reviewing clause language for clarity",
                    "alternative_wording": "",
                    "rationale": "Clear language reduces misunderstandings"
                }
            ]
    
    def _get_default_contract_suggestions(self) -> List[Dict[str, Any]]:
        """Default contract suggestions when AI generation fails"""
        return [
            {
                "priority": "HIGH",
                "category": "structure",
                "issue": "Unable to generate specific suggestions",
                "suggestion": "Conduct manual contract review with legal team",
                "alternative_wording": "",
                "rationale": "Professional review ensures comprehensive risk assessment"
            }
        ]
    
    def prioritize_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort suggestions by priority and impact"""
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        return sorted(
            suggestions,
            key=lambda x: priority_order.get(x.get('priority', 'LOW'), 1),
            reverse=True
        )

suggestion_generator = SuggestionGenerator()