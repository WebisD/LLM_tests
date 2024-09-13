from langchain_community.llms.ollama import Ollama
import pandas as pd

def countTokens(text):
    llm = Ollama(model="llama3.1")
    num_tokens = llm.get_num_tokens(text)
    print(f"Número de tokens: {num_tokens}")