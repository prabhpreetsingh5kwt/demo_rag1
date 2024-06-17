
from langchain_openai import ChatOpenAI
import pandas as pd
import os
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.agents.agent_toolkits.pandas.base import _get_functions_single_prompt
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

openai_key=os.getenv("openai_key")


# st.title("Chat with CSV Assistant")

from collections import deque

# if "messages" not in st.session_state:
#     st.session_state.messages = deque(maxlen=100)  # Store up to 100 messages


def preprocess():
    df = pd.read_excel('empty_trivia.xlsx')
    template = """You are provided with a dataframe containing trivia facts. Your task is to analyze the input questions and output the relevant trivia facts in a professional conversational style. Follow these guidelines strictly:

1. **Input Analysis**:
   - If the query is 'None', return an empty string " ".
   - Extract relevant details such as the product and state from the input question.

2. **Trivia Retrieval**:
   - Search the dataframe for trivia facts that match the extracted product and state.

3. **Output Formatting**:
   - Present the trivia facts in a professional and engaging manner. Do not use the word "Trivia".
   - Use phrases like "Did you know?" to introduce the fact.

4. **Output**:
   - Return the trivia fact exactly as it is from the dataframe, but formatted as specified.
   - If no matching trivia is found, return an empty string " ".

**Examples**:

1. **Example 1**:
   - Input: "windshield protection in Alabama?"
   - Output: "Did you know? In Alabama, the combination of high temperatures and occasional hailstorms leads to a significant number of windshield cracks and chips each year. In fact, it's estimated that over 10,000 windshields need repairs annually due to these conditions."

2. **Example 2**:
   - Input: "None"
   - Output: " "

3. **Example 3**:
   - Input: "gap in Colorado?"
   - Output: "Did you know? In Colorado, GAP insurance is highly recommended due to the state's high rate of car theft and total loss incidents. This coverage helps bridge the gap between the car's value and the remaining loan balance in such cases."

**Guidelines**:
- Ensure that the response is engaging and informative.
- Maintain a professional tone throughout.
- Always use the phrase "Did you know?" at the beginning of the trivia fact.
- If no relevant trivia fact is found, return an empty string " ".

**IMPORTANT**: **ONLY RETURN THE FINAL TRIVIA FACT OR AN EMPTY STRING " "**
"""

    prompt = _get_functions_single_prompt(df)
    prompt.input_variables.append("chat_history")
    prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
    prompt.messages.insert(2, SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],template=template)))
    tools = [
        PythonAstREPLTool(
            locals={"df": df},
            description="""
                        A Python shell. Use this to execute python commands. Input should be a valid python command.

                        Try and break the problem in steps and solve the problem step by step.

                        When the query involves an aggregated operation, first fetch the values of the relevant column,
                        and then perform the aggregated operation.

                        Examples to follow below:-
                        Question - My car is corola and 2006 model  , I want to buy a premium ,what is the price?
                        Answer -
                        <steps to solve the problem>
                        <relevant trivia if applicable>

                        When using this tool, sometimes the output is abbreviated - make sure it does not look abbreviated before
                        using it in your answer.
                        """,
        )
    ]
    chat_model = ChatOpenAI(api_key=openai_key,temperature=0.0)
    agent = create_openai_functions_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

#EXAMPLE:
# query="Tire & Wheel in alabama"
# exec=preprocess()
# response = exec.invoke({"input": query, "chat_history": []})
# print(response['output'])