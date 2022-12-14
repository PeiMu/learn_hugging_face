from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."))

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)

pt_outputs = model(**pt_batch)
print(pt_outputs)

