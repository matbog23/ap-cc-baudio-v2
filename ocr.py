import asyncio
import cv2
import os
import time
import easyocr
import requests
import re
from websockets.server import serve

from secret import OPENAI_API_KEY

# Set your OpenAI API key here
#OPENAI_API_KEY = TEMPLATE

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to analyze text sentiment
def analyzeTextSentiment(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "model": "gpt-3.5-turbo-0613",
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "messages": [
            {"role": "system", "content": "You are a system that understands the data gathered from a book using OCR, analyzes the text and then returns the predominant feeling in a single word. You can only pick from the list: happy, sad, angry or neutral."},
            {"role": "user", "content": "Please analyze the text and determine the predominant feeling. Answer in a single word from the list: happy, sad, angry or neutral." + prompt},
            {"role": "assistant", "content": "You can only answer with a single word, from the list: happy, sad, angry or neutral. write the answer in lower case."},
        ]
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        sentiment = response_json["choices"][0]["message"]["content"]
        sentiment = preprocess_text(sentiment)
        return sentiment
    else:
        print("Error:", response.status_code)
        print("Response Content:", response.content)
        return "Error analyzing text"

async def send_result(websocket, path):
    print("Client connected")

    # Capture frame-by-frame
    cap = cv2.VideoCapture(0)  # adjust, 0 = computer webcam, 1 = external webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

            # Save the image with a unique filename (timestamp)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            # Read text from the captured image
            result = reader.readtext(filename)

            # Print the detected text
            detected_text = ""
            if result:
                print("Detected Text:")
                for detection in result:
                    print(detection[1])
                    detected_text += detection[1] + " "

                print("Volledige OCR tekst: ", detected_text)

                # Generate sentiment based on detected text
                sentiment = analyzeTextSentiment(detected_text)
                print(sentiment)

                # Send the sentiment over the WebSocket connection
                await websocket.send(sentiment)

            # Wait for 10 seconds
            await asyncio.sleep(3) #await

    finally:
        # Release the capture
        cap.release()
        cv2.destroyAllWindows()

async def main():
    print("Starting server...")
    async with serve(send_result, "localhost", 8765):
        print("Server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
