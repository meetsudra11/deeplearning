from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv 
import streamlit as st 
import os 
from langchain_core.prompts import PromptTemplate,load_prompt


load_dotenv() 


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

st.header("Reasearch Tool")
paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explaination Style",["Beginner-Friendly","Technical","Code-Oriented","Mathemtical"])
length_input = st.selectbox("Select Explaination length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explaination)"])

template = load_prompt('/Users/meetsudra/Documents/GitHub/deeplearning/Langchain/template.json')

# fill the placeholders 
prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)