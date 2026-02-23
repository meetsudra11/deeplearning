from langchain_openai import OpenAI
# this is an integration of langchain with openai
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
result = llm.invoke("What is the weather today in mumbai")
print(result)