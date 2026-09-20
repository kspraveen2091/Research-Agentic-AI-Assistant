# 📰 AI News Research & Summarizer
# Problem statement

# Build an AI system that takes a topic such as "Impact of AI on software jobs", 
# researches it from multiple sources,
# evaluates the findings, and produces a concise report.


# load env
from dotenv import load_dotenv
load_dotenv()



# create state schema
from typing import TypedDict
class State(TypedDict):
    topic: str
    report_1: str
    report_2: str
    report_3: str
    detailed_report: str
    evaluation_result: str
    concise_report: str


# Create model
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="openrouter",
    temperature=0
)


# Create Nodes
def get_topic(state: State):
    return {
        "topic": state["topic"]
    }


def report_1(state: State):

    prompt = f"""
    Research the topic: {state["topic"]}

    Focus on Wikipedia
    Provide the findings in 100 words.
    """

    response = model.invoke(prompt)

    return {
        "report_1": response.content
    }


def report_2(state: State):

    prompt = f"""
    Research the topic: {state["topic"]}

    Focus on The hindu newspaper article.
    Provide the findings in 100 words.
    """

    response = model.invoke(prompt)

    return {
        "report_2": response.content
    }


def report_3(state: State):

    prompt = f"""
    Research the topic: {state["topic"]}

    Focus on The Times of India newspaper article.
    Provide the findings in 100 words.
    """

    response = model.invoke(prompt)

    return {
        "report_3": response.content
    }


def detailed_report(state: State):

    prompt = f"""
    Combine the following three research reports
    into one detailed report.

    Report 1:
    {state["report_1"]}

    Report 2:
    {state["report_2"]}

    Report 3:
    {state["report_3"]}
    """

    response = model.invoke(prompt)

    return {
        "detailed_report": response.content
    }


def evaluation(state: State):

    prompt = f"""
    You are an expert evaluator.

    Evaluate whether this detailed report contains
    sufficient information to answer the topic.

    Detailed report:
    {state["detailed_report"]}

    Return ONLY one of:

    Sufficient
    Insufficient
    """

    response = model.invoke(prompt)

    return {
        "evaluation_result": response.content.strip()
    }


def concise_report(state: State):

    prompt = f"""
    Create a concise 100-word report from:

    {state["detailed_report"]}
    """

    response = model.invoke(prompt)

    return {
        "concise_report": response.content
    }

"""from langgraph.types import Send
def routing(state: State):

    if state["evaluation_result"].strip().lower() == "sufficient":

        return "sufficient"

    else:

        return [
            Send("report_1", state),
            Send("report_2", state),
            Send("report_3", state)
        ]
        """


from typing import Literal

def routing(state: State) -> Literal[
    "concise_report",
    "report_1",
    "report_2",
    "report_3"
]:

    if state["evaluation_result"].strip().lower() == "sufficient":
        return "concise_report"

    return ["report_1", "report_2", "report_3"]

# Create Graph

from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)


# Add Nodes

builder.add_node("get_topic", get_topic)
builder.add_node("report_1", report_1)
builder.add_node("report_2", report_2)
builder.add_node("report_3", report_3)
builder.add_node("detailed_report", detailed_report)
builder.add_node("evaluation", evaluation)
builder.add_node("concise_report", concise_report)


# Create edges

builder.add_edge(START,"get_topic")
builder.add_edge("get_topic", 'report_1')
builder.add_edge("get_topic", 'report_2')
builder.add_edge("get_topic", 'report_3')

builder.add_edge('report_1','detailed_report')
builder.add_edge('report_2','detailed_report')
builder.add_edge('report_3','detailed_report')

builder.add_edge("detailed_report","evaluation")

builder.add_conditional_edges(
    "evaluation",routing
    #{"sufficient": "concise_report"}
    )

builder.add_edge("concise_report",END)


# Compile
graph = builder.compile()


# Graph visualization
import os

png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph_1.png", "wb") as f:
    f.write(png_bytes)

os.startfile("graph_1.png")


# Invoking graph
result = graph.invoke({"topic": "Origin of Life"})


# Display
print(result["concise_report"])