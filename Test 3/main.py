import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

PROMPT_TEMPLATE4 = """
From the following context, extract all numerical data and determine which of them most strongly correlates with the feature: {feature}. Provide a ranked list of the numerical data along with their correlation values or strength of association with this feature.

Context: {context}

Feature of interest: {feature}

Please provide all the following information in a single markdown table:

1. Data: The extracted numerical data.
2. Description: A description of what each numerical data point represents.
3. Correlation Strength: The correlation strength or measure of association with {feature}.
4. Context Phrase: A short phrase or sentence from the context where the numerical data was found.
5. Explanation: An explanation of why this numerical data correlates with the feature.

Output: A single markdown table in this format:

| Data           | Description               | Correlation Strength | Context Phrase             | Explanation                    |
|----------------|---------------------------|----------------------|----------------------------|--------------------------------|
| Example Data   | Description of the Data   | Example Correlation  | Phrase where it was found  | Explanation of the correlation |
"""

local_path = "../Data/Relatorio_boticario.pdf"

loader = PyPDFLoader(file_path=local_path)
data = loader.load()

context = "\n".join(page.page_content for page in data)

#llm = Ollama(model="llama3.1", num_ctx=128000)
#num_tokens = llm.get_num_tokens(context)
#print(f"Número de tokens: {num_tokens}")

feature = "Company Size"
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE4)
prompt.format(feature=feature, context=context)
llm = Ollama(model="llama3.1", num_ctx=128000)

rag_chain = prompt | llm | StrOutputParser()

response_text = rag_chain.invoke({"context": context, "feature": feature})

fname = "../Results/test3.txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write(response_text)