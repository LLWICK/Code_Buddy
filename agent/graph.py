from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b");

user_prompt = "create a simple calculator web application"

prompt = f"""

You are a project PLANNER agent . convert the user prompt into a COMPLETE engineering project plan
user request : {user_prompt}

"""


class Schema(BaseModel):
    pass
    

result = llm.with_structured_output(Schema).invoke(prompt)
print(result)