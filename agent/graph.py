from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from prompts import *
from states import *
from langgraph.graph import StateGraph, END, START
load_dotenv()
import json
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0);




user_prompt = "calculator web based app"

prompt = planner_prompt(user_prompt)

def planner_agent(state: agent_state)-> agent_state:
    print("executing PLANNER agent Node...")
    resp = llm.with_structured_output(Plan).invoke(state["messages"])

    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    
    
    return {
        "messages": [resp]
    }

def architect_agent(state: agent_state)-> agent_state:
    
    print("executing ARCHITECT agent Node...")
    plan: Plan = state["messages"][-1]

    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(state)
    )

    if resp is None:
        raise ValueError("Architect did not return a valid response.")
    
    else:
        state['messages'] = state['messages'].append(resp)
    
    return state;
    


    

#result = llm.with_structured_output(Plan).invoke(prompt)
#print(result)

builder = StateGraph(agent_state)

builder.add_node("planner_agent_node", planner_agent)
builder.add_node("architect_agent_node", architect_agent)

builder.add_edge(START, "planner_agent_node")
builder.add_edge("planner_agent_node", "architect_agent_node")
builder.add_edge("architect_agent_node", END)

graph = builder.compile()

response = graph.invoke({"messages": prompt})

print(response["messages"])
