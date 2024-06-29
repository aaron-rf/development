
### Building Our First Demo ###
# pip install gradio
import gradio as gr

def greet(name):
    return "Hello " + name

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()


import gradio as gr
def greet(name):
    return "Hello " + name

textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)
gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()


# Including Model Predictions
from transformers import pipeline
model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

print(
    predict("My favourite programming language is: ")
)

import gradio as gr
gr.interface(fn=predict, inputs="text", outputs="text").launch()
### END Building Our First Demo ###



### Understanding the Interface Class ###
# A Simple Example wih Audio
import numpy as np
import gradio as gr

def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(source="microphone", type="numpy", label="Speak here...")
gr.Interface(reverse_audio, mic, "audio").launch()

# Handling Multiple Inputs and Outputs
import numpy as np
import gradio as gr
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def generate_tone(note, octave, duration):
    sr = 48_000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20_000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)

gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Textbox(type="number", value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()


# Interface For a Speech-Recognition Model
from transformers import pipeline
import gradio as gr
model = pipeline("automatic-speech-recognition")

def transcribe_audio(mic=None, file=None):
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must either provide a mic recording or a file"
    transcription = model(audio)["text"]
    return transcription

gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
).launch()


# Sharing Demos with Others
title = "Ask Rick a Question"
description = """
The bot was training...
<img src="">
"""
article =" Checkout this URL...[descriptoin](http)"

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    exmaples=[["What are you doing?"], ["Where should we time travel to?"]],
).launch()


classify_image = ""
gr.Interface(classify_image, "image", "label").launch(share=True)



# Sketch Recognition Demo
from pathlib import Path
import torch
import gradio as gr
from torch import nn

LABELS = Path("class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(stte_dict, strict=False)
model.eval()

def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {
        LABELS[i]: v.item()
        for i, v in zip(indices, values)
    }

interface = gr.Interface(
    predict,
    inptus="sketchpad",
    outputs="label",
    theme="huggingface",
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and th ealgorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
)
interface.launch(share=True)
### END Understanding the Interface Class ###





### Integrations with the Hugging Face Hub ###
# Loading Models From the Hugging Face Hub
import gradio as gr
title = "GPT-J-6B"
description = "Gradio Demo for GPT-J 6b, ..."
article = "<p style='text-align: center'><a href='https:... target='_blank'>GPT-J-6B: A 6 Billion Paremeter Autoregressive Language Model</a></p>'"
gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
    article=article,
).launch()


# Loading From Hugging Face Spaces
gr.Interface.load("spaces/abidlabs/remove-bg").launch()

gr.Interface.load(
    "spaces/abidlabs/remove-bg", inputs="webcam", title="Remove your webcam background"
).launch()
### End Integrations with the Hugging Face Hub ###



### Advanced Interface Features ###
# Using State To Persist Data
import random
import gradio as gr

def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    
    history.append((message, response))
    return history, history

iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshow=False,
    allow_flaggign="never",
)
iface.launch()



# Using Interpretation to Understand Predictions
import requests
import tensorflow as tf
import gradio as gr
inception_net = tf.keras.applications.MobileNetV2()
response = requests.get("https://...")
labels = response.text.split("\n")

def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)
title = "Gradio Image Classification + Interpretation Example"
gr.Interface(
    fn=classify_image, inputs=image, outputs=label, interpretation="default", title=title
).launch()

### End Advanced Interface Features ###




### Introduction to Gradio Blocks ###
# Creatint a Simple Demo Using Blocks
import gradio as gr

def flip_text(x):
    return x[::-1]

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Flip Text!
    Start typing below to see the output
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()
    input.change(fn=flip_text, inuts=inputs, outputs=output)

demo.launch()

# Customizing the Layour Of Our Demo
import numpy as np
import gradio as gr
demo = gr.Blocks()

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()


# Exploring Events and State
import gradio as gr
api = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")

def complete_with_gpt(text):
    return text[:-50] + api(text[-50:])

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type here and press enter...", lines=4)
    btn = gr.Button("Generate")
    btn.click(complete_with_gpt, textbox, textbox)

demo.launch()



# Creating Multi-Step Demos
from transformers import pipeline
import gradio as gr
asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()
    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()



# Update Component Properties
import gradio as gr

def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        return gr.Textbox.update(visible=False)
    
with gr.Blocks() as block:
    radio = gr.Radio(
        ["short", "long", "none"], label="What kind of essay would you like to write?"
    )
    text = gr.Textbox(lines=2, interactive=True)
    radio.change(fn=change_textbox, inputs=radio, outputs=text)
    block.launch()


### End Introduction to Gradio Blocks ###
