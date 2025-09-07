# Create a Parallel Chain with the help of runnable--(runnable helps to create the parallel chain in LangChain)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence


load_dotenv()

# Create the model


llm1= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

llm2= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task= "text-generation", max_new_tokens=300)

llm3= HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", task="text-generation", max_new_tokens=100)

#llm= HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")

model1= ChatHuggingFace(llm=llm1)

model2= ChatHuggingFace(llm=llm2)

model3= ChatHuggingFace(llm=llm3)

# create the prompt

prompt1= PromptTemplate(template="create a short report on the following text \n {text}",
                        input_variables=["text"])

prompt2= PromptTemplate(template="create 5 shorts questions answers from the above text \n {text}",
                        input_variables=["text"])

prompt3= PromptTemplate(template="Merge the short report and quiz into a single document\n notes-->{notes} and quiz -->{quiz}",
                        input_variables=["notes", "quiz"])

# create the parser

parser= StrOutputParser()

# Create the Parallel Chain with the help of Runnable

parallel_chain= RunnableParallel({"notes": prompt1 | model1 | parser,
                                  "quiz": prompt2 | model2 | parser})

merger_chain= prompt3 | model3 | parser

final_chain= parallel_chain | merger_chain

result =final_chain.invoke({"text":"what is SVM in machine learning"})

# print the result

print(result)




