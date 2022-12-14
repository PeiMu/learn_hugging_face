from transformers import pipeline

# example of sentiment-analysis
classifier = pipeline(task="sentiment-analysis")

print(classifier("We are very happy to show you the 🤗 Transformers library."))

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

