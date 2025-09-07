
# Create a Simple Chain

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Create a prompt

prompt= PromptTemplate(template= "Generate 5 interesting fact about the {topic}",
                       input_variables=['topic'])

# Model created

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

# create the parser

parser= StrOutputParser()

# create the chain 

chain= prompt | model | parser

# Invoke the chain

result= chain.invoke({"topic":"NEP about the fees structure of school"})

# print the result

print(result)

# create the visualization of the chain

chain.get_graph().print_ascii()