from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-pro  ')
results = model.invoke("What is the weather today in mumbai")
print(results.content)