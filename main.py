import streamlit as st
from operator import itemgetter
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def create_chain(system_prompt):

    model_path = 'llama-2-7b.Q2_K.gguf'

    llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            top_p=1,
            verbose=False,
            streaming=True,
            stop=["Human:"]
            )

    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{human_input}"),
            ("ai", ""),
        ])

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    def save_memory(inputs_outputs):
        inputs = {"human": inputs_outputs["human"]}
        outputs = {"ai": inputs_outputs["ai"]}
        memory.save_context(inputs, outputs)

    def extract_response(chain_response):
        return chain_response["ai"]

    llm_chain = {
            "human_input": RunnablePassthrough(),
            "chat_history": (
                RunnableLambda(memory.load_memory_variables) |
                itemgetter("chat_history")
            )
        } | prompt | llm

    chain_with_memory = RunnablePassthrough() | {
                "human": RunnablePassthrough(),
                "ai": llm_chain
            } | {
                "save_memory": RunnableLambda(save_memory),
                "ai": itemgetter("ai")
            } | RunnableLambda(extract_response)

    return chain_with_memory


st.set_page_config(
    page_title="Your own Chat!"
)

st.header("Your own Chat!")

system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")

llm_chain = create_chain(system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Your message here", key="user_input"):

    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = llm_chain.invoke(user_prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)