from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from config.settings import settings
from backend.db import db_manager
from backend.embeddings import embedding_manager
import json
import logging

class ContractChatbot:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.conversation_history = []
    
    def chat(self, user_query: str, contract_id: Optional[int] = None) -> Dict[str, Any]:
        """Process user query and provide response with contract context"""
        try:
            # Get relevant context
            context = self._get_relevant_context(user_query, contract_id)
            
            # Generate response
            response = self._generate_response(user_query, context)
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_query,
                "assistant": response["answer"],
                "context_used": context is not None
            })
            
            return response
        except Exception as e:
            logging.error(f"Error in chatbot: {e}")
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _get_relevant_context(self, query: str, contract_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve relevant context for the query"""
        try:
            # Generate embedding for the query
            query_embedding = embedding_manager.generate_embedding(query)
            
            # Search for similar clauses
            similar_clauses = db_manager.search_similar_clauses(query_embedding, limit=5)
            
            context = {
                "similar_clauses": similar_clauses,
                "contract_details": None
            }
            
            # If specific contract is mentioned, get its details
            if contract_id:
                context["contract_details"] = db_manager.get_contract_details(contract_id)
            
            return context
        except Exception as e:
            logging.error(f"Error getting context: {e}")
            return None
    
    def _generate_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Azure OpenAI"""
        try:
            # Prepare context information
            context_text = self._format_context(context)
            
            # Determine query intent
            intent = self._classify_query_intent(query)
            
            # Create appropriate prompt
            prompt = self._create_prompt(query, context_text, intent)
            
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources from context
            sources = self._extract_sources(context) if context else []
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": self._estimate_confidence(answer, context),
                "intent": intent
            }
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return {
                "answer": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0,
                "intent": "unknown"
            }
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the user's query"""
        query_lower = query.lower()
        
        intent_patterns = {
            "risk_assessment": ["risk", "dangerous", "liability", "exposure", "safe"],
            "clause_explanation": ["what does", "explain", "meaning", "definition"],
            "comparison": ["compare", "difference", "better", "versus", "vs"],
            "suggestion": ["improve", "better", "recommend", "suggest", "fix"],
            "compliance": ["compliant", "compliance", "legal", "regulation", "standard"],
            "summary": ["summary", "summarize", "overview", "main points"],
            "search": ["find", "search", "look for", "show me"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return "general"
    
    def _create_prompt(self, query: str, context_text: str, intent: str) -> str:
        """Create appropriate prompt based on intent"""
        base_prompt = f"User Query: {query}\n\n"
        
        if context_text:
            base_prompt += f"Relevant Context:\n{context_text}\n\n"
        
        intent_instructions = {
            "risk_assessment": "Focus on identifying and explaining potential risks. Use specific examples from the context if available.",
            "clause_explanation": "Provide a clear, detailed explanation of the clause or concept. Break down legal language into plain English.",
            "comparison": "Compare the different aspects mentioned, highlighting key differences and implications.",
            "suggestion": "Provide actionable suggestions for improvement. Be specific and practical.",
            "compliance": "Focus on compliance aspects and regulatory considerations.",
            "summary": "Provide a concise but comprehensive summary of the key points.",
            "search": "Help the user find the specific information they're looking for.",
            "general": "Provide a helpful, informative response based on the available context."
        }
        
        instruction = intent_instructions.get(intent, intent_instructions["general"])
        base_prompt += f"Instructions: {instruction}\n\n"
        base_prompt += "Please provide a clear, accurate, and helpful response."
        
        return base_prompt
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context information for the prompt"""
        if not context:
            return ""
        
        context_parts = []
        
        # Add similar clauses
        if context.get("similar_clauses"):
            context_parts.append("Similar Contract Clauses:")
            for i, clause in enumerate(context["similar_clauses"][:3], 1):
                context_parts.append(f"{i}. {clause['clause_text'][:200]}...")
                context_parts.append(f"   Risk Level: {clause['risk_level']}, Type: {clause['clause_type']}")
        
        # Add contract details if available
        if context.get("contract_details"):
            contract = context["contract_details"]["contract"]
            context_parts.append(f"\nContract Information:")
            context_parts.append(f"- File: {contract['filename']}")
            context_parts.append(f"- Overall Risk: {contract.get('risk_level', 'Unknown')}")
            context_parts.append(f"- Risk Score: {contract.get('overall_risk_score', 'N/A')}")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from context"""
        sources = []
        
        if context and context.get("similar_clauses"):
            for clause in context["similar_clauses"][:3]:
                sources.append({
                    "type": "clause",
                    "text": clause["clause_text"][:100] + "...",
                    "risk_level": clause["risk_level"]
                })
        
        return sources
    
    def _estimate_confidence(self, answer: str, context: Optional[Dict[str, Any]]) -> float:
        """Estimate confidence in the response"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if context was available
        if context:
            if context.get("similar_clauses"):
                confidence += 0.3
            if context.get("contract_details"):
                confidence += 0.2
        
        # Decrease confidence for very short answers
        if len(answer.split()) < 20:
            confidence -= 0.1
        
        # Increase confidence for detailed answers
        if len(answer.split()) > 100:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the chatbot"""
        return """
        You are a contract analysis expert specializing in technology services agreements. 
        Your role is to help users understand contract clauses, assess risks, and provide practical advice.
        
        Key principles:
        - Provide accurate, helpful information based on the provided context
        - Explain legal concepts in plain English
        - Highlight potential risks and suggest improvements
        - Be specific and actionable in your recommendations
        - If you're unsure about something, say so clearly
        - Focus on practical business implications
        
        Business Context:
        - The company provides data analytics, ML deployment, and dashboard development services
        - Contracts are typically fixed-fee with milestone payments
        - IP protection and clear scope definition are critical
        - Client data confidentiality is essential while allowing anonymized insights use
        """
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        try:
            # Create a summary of the conversation
            summary_prompt = f"""
            Summarize this conversation between a user and a contract analysis assistant:
            
            {json.dumps(self.conversation_history, indent=2)}
            
            Provide a brief summary of the main topics discussed and key insights provided.
            """
            
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "Provide concise conversation summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating conversation summary: {e}")
            return "Unable to generate conversation summary."

contract_chatbot = ContractChatbot()