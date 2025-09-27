import os

import gradio as gr
from langchain.chat_models.ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import Tool, initialize_agent, AgentType

llm = None
memory = None
search = None
search_tool = None
tools = None

def chat_interface(user_input, chat_history):
    """
    user_input: str - User input
    chat_history: list of tuples - previous chat [(user, AI), ...]
    """
    response = agent.run(user_input)

    chat_history = chat_history or []
    chat_history.append((user_input, response))

    return "", chat_history


if __name__ == "__main__":
    # -------------------------
    # 1. Initialize LLM
    # -------------------------
    llm = ChatOllama(
        model="mistral",
        base_url="http://localhost:11434/",
        temperature=0.2
    )

    # -------------------------
    # 2. Set up memory
    # -------------------------
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # -------------------------
    # 3. Initialize SerpAPI wrapper
    # -------------------------
    search = SerpAPIWrapper(serpapi_api_key=os.environ["SERP_API_KEY"])

    # Wrap it as a LangChain Tool
    search_tool = Tool(
        name="Web Search",
        func=search.run,
        description="Use this tool to answer questions by searching online."
    )

    # -------------------------
    # 4. Initialize agent
    # -------------------------
    tools = [search_tool]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    with gr.Blocks() as demo:
        gr.Markdown("# Ollama Mistral Chatbot with Online Search")
        chat = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        submit_btn = gr.Button("Send")

        submit_btn.click(
            chat_interface,
            inputs=[user_input, chat],
            outputs=[user_input, chat]
        )

    demo.launch()