from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

app = Flask(__name__)

# Load model and tokenizer
model_name = "bert_problem_identification_1.0000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

# Load label mapping
df = pd.read_csv("problem_identification_dataset.csv")
label_mapping = dict(enumerate(df['Category'].astype('category').cat.categories))

# Predict function
def predict(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return label_mapping[prediction.item()], confidence.item()

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        user_input = request.form['problem']
        prediction, confidence = predict(user_input)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
