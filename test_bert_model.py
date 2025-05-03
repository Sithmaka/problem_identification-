import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load the Best Trained Model
model_name = "bert_problem_identification_1.0000"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {model_name}")
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.to(device)
model.eval()

# Load Label Mapping from Dataset
df = pd.read_csv("problem_identification_dataset.csv")
label_mapping = dict(enumerate(df['Category'].astype('category').cat.categories))

# Function for Making Predictions
def predict(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return label_mapping[prediction]

# Interactive User Input
print("\n**Problem Identification Model**")
print("Type a problem statement (or type 'exit' to quit):")

while True:
    user_input = input("\nEnter your problem statement: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    prediction = predict(user_input)
    print(f"**Predicted Category:** {prediction}")