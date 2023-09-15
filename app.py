import streamlit as st
import pandas as pd
import chardet
from langchain.callbacks.manager import Callbacks
from typing import Any, List, Tuple, Union, Dict
from json import JSONDecodeError
import json
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.chat.base import ChatAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.agent import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import SystemMessage
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.tools.python.tool import PythonAstREPLTool
from pandasql import sqldf
from langchain.agents import Tool
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent, 
    _format_intermediate_steps, 
    _FunctionsAgentAction
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
)
import os
from dotenv import load_dotenv

load_dotenv() 
# Set OpenAI API key
openai_api_key=os.environ["OPENAI_API_KEY"]
# Set OpenAI API key

# # Read the CSV file
# df = pd.read_csv("mydata.csv", encoding='cp1252')
# Streamlit app header
st.title("Pandas DataFrame Query App")


# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp1252')

class CustomPythonAstREPLTool(PythonAstREPLTool):
    name = "python"
    description = (
        "A Python shell. Use this to execute python commands. "
        "The input must be an object as follows: "
        "{'__arg1': 'a valid python command.'} "
        "When using this tool, sometimes output is abbreviated - "
        "Make sure it does not look abbreviated before using it in your answer. "
        "Don't add comments to your python code."
    )

def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_call = message.additional_kwargs["function_call"]
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            print(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )
            _tool_input = function_call["arguments"]

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)

class CustomOpenAIFunctionsAgent(OpenAIFunctionsAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.
        Returns:
            Action specifying what tool to use.
        """
        user_input = kwargs["input"]
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        prompt = self.prompt.format_prompt(
            input=user_input, agent_scratchpad=agent_scratchpad
        )
        messages = prompt.to_messages()
        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision
    

user_query = st.text_input("Enter your query:")

if uploaded_file is None:
    st.button("Enter Query", disabled=True)


else:
    if st.button("Enter Query"):
        # User input for query
        

        # Define the Streamlit app
        @st.cache_data
        def run_agent(query):
            global df
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

            prefix = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
            It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

            <df>
            {dhead}
            </df>

            You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
            You also do not have use only the information here to answer questions - you can run intermediate queries to do exploratory data analysis to give you more information as needed.

            You have a tool called `person_name_search` through which you can look up a person by name and find the records corresponding to people with similar names as the query.
            You should only really use this if your search term contains a person's name. Otherwise, try to solve it with code.But you should never return Code to User if you cant understand something   after your all approaches reply that please provide me with a clear context.

            For example:

            <question>How old is Jane?</question>
            <logic>Use `person_name_search` since you can use the query `Jane`</logic>

            <question>Who has id 320</question>
            <logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't include it.</logic>
            """
            prefix = prefix.format(dhead=df.head().to_markdown())

            dataset = {"df": df}
            tools = [CustomPythonAstREPLTool(locals=dataset)]
            tool_names = [tool.name for tool in tools]
            prompt = CustomOpenAIFunctionsAgent.create_prompt(system_message=SystemMessage(content=prefix))
            agent = AgentExecutor.from_agent_and_tools(
                agent=CustomOpenAIFunctionsAgent(llm=llm, prompt=prompt, tools=tools, verbose=True),
                tools=tools, verbose=True
            )

            result = agent.run(query)
            return result
        
        if user_query:
            st.write("Query Result:")
            result = run_agent(user_query)
            st.write(result)


