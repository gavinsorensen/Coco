import tempfile
import numpy as np
import streamlit as st
import soundfile as sf
import openai
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper

# API Keys as User Inputs instead of being hard-coded
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
set_api_key("Your ElevenLabs API key")  # Your ElevenLabs API Key
openai.api_key = openai_api_key

# Transcription function
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"].strip()

# AI Assistant function
def ai_assistant(user_input):
    llm = OpenAI(temperature=.5, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history")
    zapier = ZapierNLAWrapper(zapier_nla_api_key="Your Zapier NLA API Key")  # Your Zapier NLA API Key
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    tools = toolkit.get_tools() + load_tools(["human"])
    agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
    assistant_message = agent.run(user_input)
    return assistant_message

# Streamlit layout and interaction
st.title("AI Assistant")
audio_file = st.file_uploader("Upload Audio", type=["wav"])
text_input = st.text_input("Or type your query")
user_input = ""

if audio_file is not None:
    tempfile_path = tempfile.mkdtemp()
    audio_path = os.path.join(tempfile_path, "audio.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_file.getvalue())
    user_input = transcribe_audio(audio_path)
elif text_input:
    user_input = text_input

if user_input:
    st.text("You said: {}".format(user_input))
    st.text("Assistant reply: {}".format(ai_assistant(user_input)))
