# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI

# Initialize LLM with explicit endpoint + key
llm = AzureChatOpenAI(
    deployment_name="neostats_hackathon_api_v4",
    model="gpt-oss-120b",                         
    temperature=0,
    api_version="2024-05-01-preview",             
    api_key="4nkzckeyRZmen99IDRUgnaPx20FloouZy5Hff1U8gO7jE0glW80c03mJQQJ99BIACYeBjFXJ3w3AAAAACOGFDyO",                      # Replace with your actual key
    azure_endpoint="https://neoaihackathon.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview" 
)

# Simple call
response = llm.invoke("Hi")
print(response.content)
