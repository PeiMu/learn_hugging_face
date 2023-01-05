from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(
    [
        "This course is amazing",
    ]
)

print("classifier: ", result)

################################################################
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print("tokenizer: ", inputs)

################################################################
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print("last hidden state: ", outputs.last_hidden_state.shape)
print("auto model: ", outputs)

################################################################
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print("sequence classification: ", outputs)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print("predictions: ", predictions)
print("labeled: ", model.config.id2label)
