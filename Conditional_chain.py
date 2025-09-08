# Create a Conditional Chain with the help of runnable--(runnable helps to create the parallel chain in LangChain)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableSequence, RunnableLambda
# Import this two class for control/ structutre the output of LLMs by using pydantic
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

llm1= HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task= "text-generation", max_new_tokens=300)

llm2= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

model2= ChatHuggingFace(llm=llm2)

# create a parser

parser= StrOutputParser()

# create the pydantic object

class feedback(BaseModel):
    sentiment:Literal["positive", "negative"]= Field(description="Give the sentiment of the feedback")

parser2= PydanticOutputParser(pydantic_object=feedback)





# Create the prompt

prompt1= PromptTemplate(template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
                        input_variables=["feedback"],
                        partial_variables={"format_instruction":parser2.get_format_instructions()})


# create the chain

classifier_chain= prompt1 | model2 | parser2

#result= classifier_chain.invoke({"feedback": "This is the terrible smart phone"}).sentiment

prompt2= PromptTemplate(template="write an appropriate response to this positive feedback \n {feedback}",
                        input_variables=["feedback"])




prompt3= PromptTemplate(template="write an appropriate response to this negative feedback \n {feedback}",
                        input_variables=["feedback"])

# Test this output

#print(result)

branch_chain= RunnableBranch((lambda x:x.sentiment=="positive", prompt1 | model2 | parser),
                             (lambda x:x.sentiment=="negative", prompt3 | model2 | parser),
                             RunnableLambda(lambda x: "colud not find the sentiments"))


# final chain

final_chain= classifier_chain | branch_chain

result1= final_chain.invoke({"feedback": "This is terrible smart phone"})

print(result1)