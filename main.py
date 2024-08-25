import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType


# Define a tool to sum two numbers
def sum_numbers(query: str) -> str:
    """Returns the sum of two numbers."""
    try:
        a, b = map(float, query.split(','))
        return str(a + b)
    except:
        return "Please provide two numbers separated by a comma."

# Define a tool to convert a string to proper case
def to_proper_case(s: str) -> str:
    """Converts a string to proper case."""
    return s.title()

# Create Tool objects for each function
sum_tool = Tool(
    name="sum_numbers",
    func=sum_numbers,
    description="Sums two numbers. Usage: sum_numbers(a, b)"
)

proper_case_tool = Tool(
    name="to_proper_case",
    func=to_proper_case,
    description="Converts a string to proper case. Usage: to_proper_case(s)"
)

# Define a prompt template for understanding user queries
prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a smart assistant. Determine the task from the user's query: "{query}".
    If the task is to add numbers, use the sum_numbers tool.
    If the task is to convert a string to proper case, use the to_proper_case tool.
    """
)

# Initialize the LLM chain with the prompt and OpenAI model
llm = OpenAI(temperature=0)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Initialize the agent with the tools and the LLM chain
tools = [sum_tool, proper_case_tool]
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION , verbose=True)

# Function to run the agent
def run_agent(query):
    response = agent.run(query)
    return response

# Test the agent with different queries
queries = [
    'Please convert this STRING TO PROPER CASE.'
]

# Collect and print results for each query
results = []
for query in queries:
    results.append(run_agent(query))

print(results)