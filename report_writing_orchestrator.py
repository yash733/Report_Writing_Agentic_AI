import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.constants import Send
import operator

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

model = ChatGroq(model = "llama3-70b-8192")

# create structure output
class Section(BaseModel):
    name: str = Field(description="Report Sub heading")
    description: str = Field(description='Short and detailed overview on the topic to be covered')

class Sections(BaseModel):
    sections: list[Section] = Field(description="collection of all the sections of the report")

plan = model.with_structured_output(Sections)

#-------------------------------
class State(TypedDict):
    topic: str
    sections: list[Section]
    final_repo: str
    complete_section: Annotated[list[str], operator.add] #1

class WorkState(TypedDict):
    section: Section # defining name, description
    complete_section: Annotated[list[str], operator.add] #2 same as #1 
               # As workerState would be mutiple working parallely, and storing it in complete_section
               # 1 complete_section is same so it will also update, append data into it.
#-------------------------------

def orchestrator(state: State):
    report_sections = plan.invoke([{'role':'system', 'content':'Generate a plan for the report.'},
                 {'role':'user','content':f"Here is the report topic: {state['topic']}"}])

    return {'sections':report_sections.sections}
# sections=[
# Section(name='Introduction to Agentic AI RAGs', description='Overview of Agentic AI RAGs and their significance'), 
# Section(name='What are Agentic AI RAGs?', description='Definition and explanation of Agentic AI RAGs'), 
# ---

def condition(state: State):
    '''work_assign'''
    # Kick off section writing in parallel via Send() API
    return [Send('llm_call',{'section':sub_sections}) for sub_sections in state['sections']]

def llm_call(work_state: WorkState):
    data = model.invoke([
        {'role':'system', 'content':'Write a section of report following the provided name and description. Include no preamble for each section. Use markdown formatting.'},
        {'role':'user','content':f"Here is the section name: {work_state['section'].name} and description: {work_state['section'].description}"}])
    
    return {'complete_section':[data.content]} # passing List to .add more section from other parallel nodes.

def compile(state: State):
    data = state['complete_section']
    formated_data = '\n\n---\n\n'.join(data)

    return {'final_repo': formated_data}

# ------------------------------

graph = StateGraph(State)
graph.add_node(orchestrator)
graph.add_node(llm_call)
graph.add_node(compile)

graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator',condition,['llm_call'])
graph.add_edge('llm_call','compile')
graph.add_edge('compile',END)

agent = graph.compile()

res = agent.invoke({"topic": "Create a report on Agentic AI RAGs"})
print(res['final_repo'])
# from IPython.display import Markdown
# Markdown(res["final_repo"])
#--------------------------------
chart = agent.get_graph().draw_mermaid_png()
flowchart = './langgraph_workflow/report_orchestrator.png'
with open(flowchart, 'wb') as f:
    f.write(chart)

with open('./langgraph_workflow/report_orchestrator.txt','w') as f:
    f.write(res['final_repo'])
    
from PIL import Image
Image.open(flowchart).show()
