from transformers import pipeline

# example of sentiment-analysis
classifier = pipeline(task="sentiment-analysis")

print(classifier("We are very happy to show you the 🤗 Transformers library."))

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)
