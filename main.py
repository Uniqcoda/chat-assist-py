import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from retriever import retriever

load_dotenv()

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if environment variables are set
if not openai_api_key:
    raise ValueError("OpenAI environment variables not set. Please set them in the .env file")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize model
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

### Prompt templates

# Contextualize question
question_prefix = (
    "Given a chat history (if any) and the latest user question, "
    "formulate a standalone question. "
    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_prefix),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_question_prompt
)

# Answer question
answer_prefix = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context and the chat history to answer the question. "
    "Try to find the answer in the context. If the answer is not given in the context, find the answer in the chat history if possible."
    "If you don't know the answer, say \"I'm sorry, I don't know the answer to that.\""
    "Keep the answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_prefix),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Initialize chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Use Streamlit session state for chat history ###
def convert_to_chat_message_history(session_history) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    for message in session_history:
        if message["type"] == "human":
            chat_history.add_user_message(message["content"])
        else:
            chat_history.add_ai_message(message["content"])
    return chat_history

def main():
    st.set_page_config(page_title="LyfeGym-Bot")
    st.title("Sam from LyfeGym")
    st.info("Hello! I am Samantha. I am here to answer any question you may have about the gym and our services. How may I help you today?")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["type"]):
            st.markdown(message["content"])
    
    # Get user input
    user_input = st.text_input("Start typing...", key="user_input")
    
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append({"type": "human", "content": user_input})
            
            # Convert session state chat history to ChatMessageHistory
            chat_history = convert_to_chat_message_history(st.session_state.chat_history)
            
            # Perform question answering
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda _: chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "abc123"}}
            )
            
            answer = response["answer"]
            st.session_state.chat_history.append({"type": "ai", "content": answer})
            
            # Re-run the Streamlit app to reflect changes
            st.rerun()

if __name__ == "__main__":
    main()