import pandas as pd
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEMPLATE = """
From the following context, extract all numerical and categorical data and determine which of them most strongly correlates with the feature: {feature}. Provide a ranked list of the numerical and categorical data along with their correlation values or strength of association with this feature.

Context: {context}

Feature of interest: {feature}

Please provide all the following information in a single markdown table:

1. Data: The value of the extracted numerical or categorical data.
2. Description: A description of what each data point represents.
3. Correlation Strength: The correlation strength or measure of association with {feature}.
4. Context Phrase: A short phrase or sentence from the context where the data was found.
5. Explanation: An explanation of why this data correlates with the feature.

Output format:

| Data                    | Description               | Correlation Strength | Context Phrase             | Explanation                    |
|-------------------------|---------------------------|----------------------|----------------------------|--------------------------------|
| Value of Example Data   | Description of the Data   | Example Correlation  | Phrase where it was found  | Explanation of the correlation |

"""

PROMPT_TEMPLATE2 = """
From the following context, extract all numerical data and determine which of them most strongly correlates with the feature: {feature}. Provide a ranked list of the numerical data along with their correlation values or strength of association with this feature.

Context: {context}

Feature of interest: {feature}

Please provide all the following information in a single markdown table:

1. Data: The extracted numerical data.
2. Description: A description of what each numerical data point represents.
3. Correlation Strength: The correlation strength or measure of association with {feature}.
4. Context Phrase: A short phrase or sentence from the context where the numerical data was found.
5. Explanation: An explanation of why this numerical data correlates with the feature.

Output format:

| Data           | Description               | Correlation Strength | Context Phrase             | Explanation                    |
|----------------|---------------------------|----------------------|----------------------------|--------------------------------|
| Example Data   | Description of the Data   | Example Correlation  | Phrase where it was found  | Explanation of the correlation |
"""

PROMPT_TEMPLATE3 = """
From the following context, extract all numerical and categorical data and determine which of them most strongly correlates with the feature: {feature}. Focus on the attributes listed below, but also consider other relevant data if present.

Attributes: {attributes}

Feature of interest: {feature}

Context: {context}

Please provide all the following information in a single markdown table:

1. Data Value: The value of the extracted numerical or categorical data.
2. Description: A description of what each data point represents.
3. Correlation Strength: The correlation strength or measure of association with {feature} or the listed attributes.
4. Context Phrase: A short phrase or sentence from the context where the data was found.
5. Explanation: An explanation of why this data correlates with the feature or features.

| Data Value    | Description               | Correlation Strength  | Context Phrase                 | Explanation                      |
|---------------|---------------------------|-----------------------|--------------------------------|----------------------------------|
| Example Data  | Description of the Data   | Example Correlation   | Phrase where it was found      | Explanation of the correlation   |
"""

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

PROMPT_TEMPLATE5 = """
From the following context only, extract numerical data that most correlates with the feature: {feature}. Provide a ranked list of the numerical data along with their correlation values or strength of association with this feature.

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

# Open context file
df = pd.read_parquet('../Data/Test_maga.parquet')
context = df['text'][0]

# Feature 
feature = "Company Size"
#attributes = "Number of Employees, Annual Revenue, Number of Offices/Locations, Type of Businesses"

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE4)
#prompt.format(feature=feature, attributes=attributes, context=context)

prompt.format(feature=feature, context=context)

llm = Ollama(model="llama3.1", num_ctx=128000)

rag_chain = prompt | llm | StrOutputParser()

# Call llm
#response_text = rag_chain.invoke({"context": context, "feature": feature, "attributes":attributes})
response_text = rag_chain.invoke({"context": context, "feature": feature})

#print(response_text)

fname = "../Results/test1.txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write(response_text)