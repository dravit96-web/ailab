from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms import Ollama
from RAG import RAG_system
import subprocess
import os


class DocState(dict):
    input: str
    rag_output: str
    validation_agent: str
    final_output: str
    execution_result: str

#Agent 1 : RAG Agent

def rag_agent(state):
    input = state['input']
    result = RAG_system(input)
    state['rag_output'] = result
    print("RAG_Output:",result)
    return state

def validation_agent(state):
    rag_output = state['rag_output']
    question = state['input']
    prompt = ChatPromptTemplate.from_template(
      """You are an expert AI assistant. Review the following RAG output for accuracy and completeness.
        If the output is satisfactory, respond with 'VALID'. If there are issues, provide constructive feedback on what needs to be improved.
        RAG Output: {rag_output}
        question: {question}"""
    )
    llm = Ollama(model="gpt-oss:120b-cloud", temperature=0.3)
    response = llm.predict_messages(prompt.format_messages(rag_output=rag_output, question=question))
    state['validation_agent'] = response.content
    print("Validation Result:",response.content)
    return state


graph = StateGraph(DocState)
graph.add_node("RAG_State",rag_agent)
graph.add_node("Validation_state",validation_agent)

graph.set_entry_point("RAG_State")
graph.add_edge("RAG_State","validation_state")
graph.add_edge("validation_state",END)


app= graph.compile()

if __name__ == "__main__":
    question = input("Please enter your question: ")
    result = app.invoke({'input': question})
    
