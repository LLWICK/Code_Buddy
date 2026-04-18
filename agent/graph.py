from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from prompts import *
from states import *
from langgraph.graph import StateGraph, END, START
from langchain_core.globals import set_verbose, set_debug
from langchain.agents import create_agent
load_dotenv()
import json
from tools import *
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0);

#set_debug(True)
#set_verbose(True)


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
        architect_prompt(plan)
    )

    if resp is None:
        raise ValueError("Architect did not return a valid response.")
    
    resp.plan = plan
    
    return {
        "messages": [resp]
    }


def coder_agent(state: agent_state)-> agent_state:
    
    print("executing CODER agent Node...")
    steps = state["messages"][0].implementation_steps

    current_step_indx = 0

    current_task = steps[current_step_indx]

    existing_content = read_file.run(current_task.filepath)

    user_prmpt = (
        
            f"Task:{current_task.task_description}\n"
            f"File:{current_task.filepath}\n"
            f"Existing content: {existing_content}"
            "User write_file(path, content)  to save your changes"
    
        
    )

    system_prompt = coder_system_prompt()
    Tools = [read_file, write_file, list_files, get_current_directory]

    coderAgent = create_agent(model=llm, tools=Tools)

    coderAgent.invoke({"messages":[{"role":"system", "content":system_prompt},
                                   {"role":"user", "content":user_prmpt}]})

    
    

    """ if resp is None:
        raise ValueError("Architect did not return a valid response.") """
    
    
    
    return {
        
    }
    


    

#result = llm.with_structured_output(Plan).invoke(prompt)
#print(result)

builder = StateGraph(agent_state)

builder.add_node("planner_agent_node", planner_agent)
builder.add_node("architect_agent_node", architect_agent)
builder.add_node("coder_agent_node", coder_agent)

builder.add_edge(START, "planner_agent_node")
builder.add_edge("planner_agent_node", "architect_agent_node")
builder.add_edge("architect_agent_node", "coder_agent_node")
builder.add_edge("coder_agent_node", END)

graph = builder.compile()



if __name__ == "__main__":
    response = graph.invoke({"messages": prompt})
    print(response["messages"])
