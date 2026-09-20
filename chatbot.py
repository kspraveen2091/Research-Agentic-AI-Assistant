from dotenv import load_dotenv
from typing import TypedDict


from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# 1. Load API keys
load_dotenv()


# 2. Create LLM
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="openrouter",
    temperature=0
)


# 3. Create state
class State(TypedDict):
    message: str
    response:str


# 4. Create NOde
def chatbot(state):
    response = model.invoke(state["message"])

    return {"message": state["message"], "response": response.content}


# 5. Create Graph
builder = StateGraph(State)


# 6. Add Node
builder.add_node("chatbot", chatbot)


# 7. Add Edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# 8. Complile Graph
graph = builder.compile()


# 9. Invoke Graph
result = graph.invoke({"message": "Business opportunities of learning Langgraph?"})


# 10. Print Response
print(result["response"])