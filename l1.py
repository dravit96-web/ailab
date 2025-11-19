from langchain.llms import ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ollama(model="llama2", temperature=0.3)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""you are linux technical assistant with expertise in system administration 
    and troubleshooting provide detailed and accurate answers to the following questions, If it pertains to any other topic 
    otherwise strictly bond with "I can can only answer linux related questions.'
    question: {question}"""
)

chain = LLMChain(llm=llm, prompt=prompt)

print("CHAT BOT")

while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break 
    response = chain.run(question=question)
    print(response)