import os
from openai import OpenAI
from langchain_openai import ChatOpenAI  # 또는 openai.ChatCompletion 직접 사용

#============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # legalAIzero 폴더 경로
EMB_FILE = os.path.join(BASE_DIR, "data", "output_chunks_with_embeddings.json")

#============================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = "당신은 법률 전문가 AI입니다. 질문에 대해 정확하고 간결하게 답변하세요."
Generate_llm = OpenAI()

#============================
Ragas_llm = ChatOpenAI(model="gpt-4", temperature=0)
