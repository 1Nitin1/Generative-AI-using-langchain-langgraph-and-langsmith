from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    template="""You are an expert academic assistant. Your task is to summarize and explain the paper titled: {paper_name}.

Requirements:
- Output style: {output_style}.
- Output length: {output_length}.
- Include detailed information, key findings, and relevant mathematical equations if present in the paper.
- If the paper contains formulas or derivations, present them clearly using LaTeX formatting.
- If you do not know the content of the paper or cannot access it, do NOT invent or hallucinate details. Instead, respond with: "I do not have enough information about this paper to provide a summary."

Final instruction:
Always prioritize accuracy and clarity. Do not fabricate content. Only summarize based on actual knowledge of the paper.
""",
    input_variables=["paper_name","output_style","output_length"]
)

prompt = template.invoke({
    "paper_name":paper_input,
    "output_style":style_input,
    "output_length":length_input
})

if st.button('Summarize'):
    res = model.invoke(prompt)
    st.write(res.content)