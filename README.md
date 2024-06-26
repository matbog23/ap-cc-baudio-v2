# ap-cc-baudio-v2

_the (messy) repository for v1 can be found here:_ [_ap-cc-baudio_](https://github.com/matbog23/ap-cc-baudio)

Hey everyone!

Exciting news – I've completed my Creative Coding project! 🎉 Here's the gist: I used a camera to capture live video of a book while flipping through its pages. Then, I implemented Optical Character Recognition (OCR) to extract text in real-time. Next, I analyzed the sentiment of the text using an API. Finally, I added background music based on the sentiment to enhance the reading experience.

In short, my project involved capturing video, extracting text with OCR, analyzing sentiment, and enhancing the experience with music. It was quite the technical challenge, but I'm thrilled with the results!

[Watch the installation video](https://youtu.be/EOi6LgYESd8)

## Instructable

<details>
  <summary>Setup</summary>

Follow these steps to set up the project:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/ap-cc-baudio-v2.git
   cd ap-cc-baudio-v2
   ```

2. **Insert API Key**

   ```
   locate secret.example.py
   rename the file to secret.py
   insert your Openai API key
   ```

3. **Install Dependencies**

   ```sh
   npm install
   ```

4. **Start Local Server**

   ```sh
   npx http-server
   ```

5. **Run Software**

   ```sh
   py ocr.py
   ```

6. **Open browser**

   ```
   http://localhost:8080 or the port specified by `http-server`
   And reload the page so the console shows 'client connected'
   ```

7. **Start the process**

   ```
   Press the button on the webpage to begin.
   ```

</details>
<details>
  <summary>Code Explained</summary>
  
### Camera Capture
Using EasyOCR we are capable of using any camera connected to the computer for taking pictures. We take a picture every ten seconds (time is set using a sleep function).
```python
# Capture frame-by-frame
cap = cv2.VideoCapture(0)  # adjust, 0 = computer webcam, 1 = external webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam is opened correctly

if not cap.isOpened():
print("Error: Could not open webcam")
exit()

```
Here is an example of how the camera could be set up to read the book.
![Example Camera Setup](Docs/LucasListening.jpg)

### OCR
Then we run OCR or Optical Character Recognition over the images that we collect from the camera.
```Python
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
```
The OCR would be run over an image such as this:
![ExampleOCR](Docs/AnExmapleImage.jpg)

### Gathering the sentiment
To get the sentiment of the text we use Openai API - [ChatGPT](https://api.openai.com/v1/chat/completions)
Note, in the analyzeTextSentiment function we are passing trough the variable prompt being the result from the OCR.
```Python
# Generate sentiment based on detected text
                sentiment = analyzeTextSentiment(detected_text)
                print(sentiment)
```
```Python
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
            {"role": "system", "content": "You are a system that understands the data gathered from a book using OCR, analyzes the text and then returns the predominant feeling in a single word. You can only pick from the list: happy, sad, angry, excited, confused or neutral. Keep in mind that you are reading children books so the feelings are not too complex, you are allowed to predict the feeling that the user would feel in certain situations. Just make sure to pick from the given list."},
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
```
### Websocket
We then send that data over a websocket from our Python script to our index.html page (+JS).
```Python
 # Send the sentiment over the WebSocket connection
                await websocket.send(sentiment)
```
```JavaScript
 const socket = new WebSocket("ws://localhost:8765");

      socket.onopen = function (event) {
        console.log("WebSocket connection established.");
      };

      // Handle WebSocket message
      socket.onmessage = async function (event) {
        const sentiment = event.data;
        console.log("Received sentiment from server:", sentiment);
      }
```

### ToneJS
Using Tone.js we play the fitting music based on sentiment.
```JavaScript
 // Define audio file paths for different moods
      const audioFiles = {
        angry: "https://matbog23.github.io/ap-cc-audiofiles/angry.mp3",
        sad: "https://matbog23.github.io/ap-cc-audiofiles/sad.mp3",
        happy: "https://matbog23.github.io/ap-cc-audiofiles/happy.mp3",
        excited: "https://matbog23.github.io/ap-cc-audiofiles/confused.mp3",
        confused: "https://matbog23.github.io/ap-cc-audiofiles/excited.mp3",
        neutral: "https://matbog23.github.io/ap-cc-audiofiles/neutral.mp3",
      };
```
```JavaScript
 function playMusic(mood) {
        // Check if the mood is different from the current player
        if (mood !== "" && mood !== currentMood) {
          // Stop the current player if it exists
          if (currentPlayer !== null) {
            currentPlayer.stop();
          }

          const audioFile = audioFiles[mood];
          const player = new Tone.Player(audioFile).toDestination();
          player.autostart = true;
          player.loop = true; // Enable looping for new player

          // Update the current player and mood
          currentPlayer = player;
          currentMood = mood;
        } else if (
          mood === currentMood &&
          currentPlayer !== null &&
          !currentPlayer.state
        ) {
          // If the mood remains the same and the player exists but stopped,
          // restart it
          currentPlayer.start();
        }
      }
```

</details>
<details>
  <summary>Learnings and Takeaways</summary>

This project has once again thought me a lot. Not only have I now made my first larger Python project, I've also learned how websockets work to send data, how I can combine hardware and software, problemsolving using databases and AI, and let's not forget about pushing myself on a project made entirly from scratch. I've gone trough all the steps of the design and development process under the careful eye of my professors. This is trully a remarkable experience that any student should wish for.

My biggest issue in this project came with having to settle for less then my initial goal. In the beginning I also wanted to implement Eye tracking on the pages and have AI generated audio rather then an audio library. Although if I take another shot at this proejct I genuinly believe that we can push it even further. I look forward to what the future brings me and to what awesome projects I can build next.
</details>
