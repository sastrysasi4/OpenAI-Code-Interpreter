import os
from dotenv import load_dotenv
load_dotenv()
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.tools import Tool


def main():
    print("Start...")

    # creating a python agent
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # to run python agent
    # python_agent_executor.run(
    #     """ generate and save in current working directory 5 QRcodes
    #                            that points to www.udemy.com/course/langchain, you have qrcode package installed already"""
    # )

    # creating csv agent
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # to run csv agent
    # csv_agent.run('how many columns are there in file episode_info.csv')
    # csv_agent.run*_('In file episode_info, which writer worote the most episodes? how many episodes did he write?')
    ## the results may be less accurate

    # Grand agent will decide which agent should call
    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                              returning the results of the code execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    grand_agent.run(
        """generate and save in current working directory 5 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    )

    # grand_agent.run("print seasons ascending order of the number of episodes they have")


if __name__ == "__main__":
    main()
