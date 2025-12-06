import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models.ollama import ChatOllama
from langchain.memory import ChatMessageHistory

# 1. Initialize the Ollama LLM
conversation = None

# 3. Function to handle a single chat turn
def chat_interface(user_input, chat_history):
    """
    user_input: str - The input from the user
    chat_history: list of tuples - previous chat history [(user, AI), ...]
    """
    global conversation
    # Get AI response
    response = conversation.run(user_input)

    # Update Gradio chat history
    chat_history = chat_history or []
    chat_history.append((user_input, response))

    return "", chat_history

if __name__ == "__main__":

    # Initialize ChatOllama
    llm = ChatOllama(
        model="mistral",
        base_url="http://localhost:11434/",
        temperature=0.2
    )

    # Set up conversation memory
    history = ChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=history)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

    # Build Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Ollama Mistral Chatbot with Memory")

        chat = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        submit_btn = gr.Button("Send")

        submit_btn.click(
            chat_interface,
            inputs=[user_input, chat],
            outputs=[user_input, chat],
        )

    demo.launch()