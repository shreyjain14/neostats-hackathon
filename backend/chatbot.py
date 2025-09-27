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
            "risk_assessment": "Focus on identifying and explaining potential risks to our company, Hari and Winston Associates LLC.",
            "clause_explanation": "Provide a clear, detailed explanation of the clause or concept from our perspective.",
            "comparison": "Compare the different aspects mentioned, highlighting which is better for us.",
            "suggestion": "Provide actionable suggestions for improvement that benefit our company.",
            "compliance": "Focus on compliance aspects and regulatory considerations relevant to our business.",
            "summary": "Provide a concise summary of the key points relevant to our interests.",
            "search": "Help the user find the specific information they're looking for.",
            "general": "Provide a helpful, informative response based on the available context, keeping our company's interests in mind."
        }
        
        instruction = intent_instructions.get(intent, intent_instructions["general"])
        base_prompt += f"Instructions: {instruction}\n\n"
        base_prompt += "Please provide a clear, accurate, and helpful response that protects the interests of Hari and Winston Associates LLC."
        
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

contract_chatbot = ContractChatbot()