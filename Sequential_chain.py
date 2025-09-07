
# Create a Sequential Chain

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Create a  1st prompt

prompt1= PromptTemplate(template= "Generate the detailed report on the {topic}",
                       input_variables=['topic'])

# Create a  2nd prompt
prompt2= PromptTemplate(template= "Generate the 5 pointer summary from the text \n {text}",
                        input_variables=["text"])

# Model created

llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model= ChatHuggingFace(llm=llm)

# create the parser

parser= StrOutputParser()

# Create the chain

chain= prompt1 | model | parser | prompt2 | model | parser

# invoke the chain

result= chain.invoke({"topic":"Employment rate of India"})

# print the result

print(result)

# show the steps in the chain through the visualization

chain.get_graph().print_ascii()