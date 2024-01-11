from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
import streamlit as st
from streamlit_chat import message
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
os.getenv('OPEN_API_KEY')


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1.2, openai_api_key= os.getenv('OPEN_API_KEY'))

# history=FileChatMessageHistory('Chat_history.jason')
memory = ConversationBufferMemory(
    memory_key='chat_history',
    # chat_memory=history,
    return_messages=True
)


prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessage(content='You are a chatbot having a conversation.'),
        MessagesPlaceholder(variable_name='chat_history'), #where the memory will be stored
        HumanMessagePromptTemplate.from_template('{content}')
                      ]
)


chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)


    


#----------------------------------------------------------------------------------------------------------------------------


st.set_page_config(
    page_title='Your Custom Assistant',
    page_icon='🤖',  # Custom favicon, you can use a URL or an emoji
    layout='wide',   # Choose 'wide' or 'centered' layout
    initial_sidebar_state='expanded'  # Choose 'auto', 'expanded', or 'collapsed'
)

st.subheader('ChatGPT Assistant With Memory Using Langchain 🤖')
# chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, openai_api_key= os.getenv('OPEN_API_KEY'))

if 'messages' not in st.session_state:
  st.session_state.messages = []

with st.sidebar:
  st.image('chatbot.png', width=200)
  api_key = st.text_input('OpenAI API Key:', type='password')
  if api_key:
    os.environ['OPENAI_API_KEY']= api_key
  system_message = st.text_input(label = 'System Role 🤖')



content = st.chat_input('Send A Message')

if system_message:
    if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
       st.session_state.messages.append(SystemMessage(content=system_message))

if content:
    st.session_state.messages.append(HumanMessage(content=content))

    with st.spinner('working on your request...'):
     response = chain.run(st.session_state.messages)
    #  response = chain.run({'content': content})

    st.session_state.messages.append(AIMessage(content=str(response)))

if len(st.session_state.messages) >=1:
  if not isinstance(st.session_state.messages[0], SystemMessage):
    st.session_state.messages.insert(0, SystemMessage(content='You are a chatbot having a conversation.'))



for i,msg in enumerate(st.session_state.messages[1:]):
  if i%2 ==0:
    message(msg.content, is_user=True, key=f'{i}+ 😂')
  else:
    message(msg.content, is_user=False, key=f'{i} + 🤖')

