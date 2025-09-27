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
            context = self._get_relevant_context(user_query, contract_id)
            response = self._generate_response(user_query, context)
            
            if response.get("intent") != "error":
                self.conversation_history.append({
                    "user": user_query,
                    "assistant": response["answer"],
                    "context_used": context is not None
                })
            
            return response
        except Exception as e:
            logging.error(f"FATAL Error in chatbot chat function: {e}")
            return {
                "answer": f"I'm sorry, a critical error occurred. Please try again. \n\n**Debug Info:** {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "intent": "error"
            }
    
    def _get_relevant_context(self, query: str, contract_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve relevant context for the query"""
        try:
            query_embedding = embedding_manager.generate_embedding(query)
            similar_clauses = db_manager.search_similar_clauses(query_embedding, limit=3)
            
            context = {
                "similar_clauses": similar_clauses,
                "contract_details": None
            }
            
            if contract_id:
                context["contract_details"] = db_manager.get_contract_details(contract_id)
            
            return context
        except Exception as e:
            logging.error(f"Error getting context: {e}")
            return None
    
    def _generate_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Azure OpenAI with improved error handling and prompt structure."""
        try:
            context_text = self._format_context(context)
            intent = self._classify_query_intent(query)
            
            # The system prompt now contains all instructions and context
            system_prompt = self._get_system_prompt(context_text, intent)
            
            # The user message is now just the clean user query
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            sources = self._extract_sources(context) if context else []
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": self._estimate_confidence(answer, context),
                "intent": intent
            }
        except Exception as e:
            # THIS IS THE KEY FIX: Return the actual error message to the UI
            logging.error(f"Error generating response: {e}")
            error_message = f"An error occurred while communicating with the AI service: {str(e)}"
            return {
                "answer": f"I apologize, but I couldn't generate a proper response. Please try rephrasing your question.\n\n**Debug Info:** {error_message}",
                "sources": [],
                "confidence": 0.0,
                "intent": "error"
            }
    
    def _get_system_prompt(self, context_text: str, intent: str) -> str:
        """Get a comprehensive system prompt including context and instructions."""
        
        base_prompt = """
        You are a contract analysis expert working for Hari and Winston Associates LLC.
        Your role is to help users understand contract clauses, assess risks to our company, and provide practical advice that protects our interests.
        Always respond from the perspective of Hari and Winston Associates LLC.
        """

        intent_instructions = {
            "risk_assessment": "Focus on identifying and explaining potential risks to our company, Hari and Winston Associates LLC.",
            "clause_explanation": "Provide a clear, detailed explanation of the clause or concept from our perspective.",
            "comparison": "Compare the different aspects mentioned, highlighting which is better for us.",
            "suggestion": "Provide actionable suggestions for improvement that benefit our company.",
            "compliance": "Focus on compliance aspects and regulatory considerations relevant to our business.",
            "summary": "Provide a concise summary of the key points relevant to our interests.",
            "search": "Help the user find the specific information they're looking for.",
            "general": "Provide a helpful, informative response based on the available context, keeping our company's interests in mind."
        }
        
        # Add specific instructions based on intent
        instruction = intent_instructions.get(intent, intent_instructions["general"])
        full_prompt = f"{base_prompt}\n\n## Your Task:\n{instruction}"

        # Add the relevant context if it exists
        if context_text:
            full_prompt += f"\n\n## Relevant Context to Use:\n{context_text}"
            
        return full_prompt

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
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context information for the prompt"""
        if not context: return ""
        context_parts = []
        if context.get("similar_clauses"):
            context_parts.append("### Similar Contract Clauses:")
            for i, clause in enumerate(context["similar_clauses"], 1):
                context_parts.append(f"{i}. Clause Text: \"{clause['clause_text'][:200]}...\"")
                context_parts.append(f"   - Risk Level: {clause['risk_level']}, Type: {clause['clause_type']}")
        if context.get("contract_details"):
            contract = context["contract_details"]["contract"]
            context_parts.append(f"\n### Current Contract Information:")
            context_parts.append(f"- File: {contract['filename']}")
            context_parts.append(f"- Overall Risk to Us: {contract.get('risk_level', 'Unknown')}")
        return "\n".join(context_parts)
    
    def _extract_sources(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from context"""
        if not context or not context.get("similar_clauses"): return []
        sources = []
        for clause in context["similar_clauses"]:
            sources.append({"type": "clause", "text": clause["clause_text"][:100] + "...", "risk_level": clause["risk_level"]})
        return sources
    
    def _estimate_confidence(self, answer: str, context: Optional[Dict[str, Any]]) -> float:
        """Estimate confidence in the response"""
        confidence = 0.5
        if context:
            if context.get("similar_clauses"): confidence += 0.3
            if context.get("contract_details"): confidence += 0.2
        if len(answer.split()) < 20: confidence -= 0.1
        if len(answer.split()) > 100: confidence += 0.1
        return min(max(confidence, 0.0), 1.0)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history: return "No conversation history available."
        try:
            summary_prompt = f"Summarize this conversation between a user and a contract analysis assistant from Hari and Winston Associates LLC:\n\n{json.dumps(self.conversation_history, indent=2)}\n\nProvide a brief summary of the main topics discussed."
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