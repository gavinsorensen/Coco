from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
import tempfile
import os
import pyaudio
import subprocess
import tweepy
import openai

from elevenlabs import generate, set_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper

set_api_key("f0cf5f9744549c8a759f2ed6c20f4868")
openai.api_key = None  # Placeholder for user API key

# Set recording parameters
fs = 44100  # sample rate
channels = 1  # number of channels

recording = None  # Global variable to hold the recording data

app = Flask(__name__)

def start_recording():
    global recording
    print("Recording started.")
    frames_per_buffer = int(fs / 10)  # Buffer size
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=fs, input=True, frames_per_buffer=frames_per_buffer)
    recording = []
    while True:
        data = stream.read(frames_per_buffer)
        recording.append(np.frombuffer(data, dtype=np.int16))
        if len(recording) > 10:  # Adjust the condition based on your recording length requirement
            break
    stream.stop_stream()
    stream.close()
    audio.terminate()
    recording = np.concatenate(recording)

def transcribe_audio(recording, fs, api_key):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe(api_key, audio_file)
        os.remove(temp_audio.name)
    return transcript["text"].strip()

def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    audio_file = "output.wav"  # Provide a temporary filename
    with open(audio_file, "wb") as f:
        f.write(audio)
    ffmpeg_path = "/usr/bin/ffplay"  # Path to ffplay executable on your EC2 instance
    subprocess.run([ffmpeg_path, "-nodisp", "-autoexit", audio_file])
    os.remove(audio_file)

class TweeterPostTool(BaseTool):
    name = "Twitter Post Tweet"
    description = "Use this tool to post a tweet to Twitter."

    def _run(self, text: str) -> str:
        """Use the tool."""
        return client.create_tweet(text=text)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")

if __name__ == '__main__':
    consumer_key = "b0FKZUdCcGlaZzBaQ1g0Sk9nbE46MTpjaQ"
    consumer_secret = "NLi3Ha98T-uRYUO3BYGVyTVrLSCHQ44LTsSmIUc9RzUp-GlgYW"
    access_token = "1654255204324511744-vAjzHFO2OhGBEjDntOO1EZjxizXDLV"
    access_token_secret = "WjzVLEeu6K4yuyR9iiRMYdgPmyr6b3Z3Kf4fJ6QL7ZrdD"

    client = tweepy.Client(
        consumer_key=consumer_key, consumer_secret=consumer_secret,
        access_token=access_token, access_token_secret=access_token_secret
    )

    llm = OpenAI(temperature=0.5, openai_api_key=None)  # Placeholder for user API key
    memory = ConversationBufferMemory(memory_key="chat_history")

    zapier = ZapierNLAWrapper(zapier_nla_api_key="sk-ak-YCuaM5ejv9o0aF1m6sZLczfGbx")
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    tools = toolkit.get_tools() + load_tools(["human"])

    agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

    @app.route("/api/v1/predict", methods=["POST"])
    def predict():
        data = request.json
        api_key = data.get('api_key')
        user_input = data.get('user_input')

        if api_key and user_input:
            global recording
            if recording is None:
                start_recording()
            message = transcribe_audio(recording, fs, api_key)
            print(f"You: {message}")
            assistant_message = agent.run(message)
            play_generated_audio(assistant_message)
            recording = None  # Reset the recording variable for the next iteration

            return jsonify({"result": assistant_message})

        return jsonify({"error": "Invalid API key or user input"})

    app.run(host='0.0.0.0', port=5000)
