import json
import sys
import logging
import re
from html import unescape
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import emoji

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the tokenizer and model
MODEL_PATH = r"C:\Users\stace\gmail-job-tracker-be\model\distilbert_email_classifier"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory does not exist: {MODEL_PATH}")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Preprocessing Function
def preprocess_email(subject, body):
    content = f"{subject} {body}".lower()
    content = emoji.demojize(content)  # Convert emojis to text
    content = unescape(content)  # Handle HTML entities
    content = re.sub(r"<[^>]+>", "", content)  # Remove HTML tags
    content = re.sub(r"http\S+|www\S+", "", content)  # Remove URLs
    content = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", content)  # Remove special characters
    content = re.sub(r"\s+", " ", content).strip()  # Normalize whitespace
    return content



def classify_email(email):
    """
    Classify an email into predefined categories.
    """
    # Preprocess content
    content = preprocess_email(email.get('subject', ''), email.get('body', ''))

    # Encode for DistilBERT
    encoding = tokenizer(
        content,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    inputs = {key: val.to(device) for key, val in encoding.items()}

    # Predict category
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    categories = ['Applied', 'Interviewed', 'Offers', 'Rejected', 'Irrelevant']
    return categories[prediction]

def main():
    # Load test_emails.json
    TEST_EMAILS_PATH = r"C:\Users\stace\gmail-job-tracker-be\test_emails.json"  # Adjust the path if necessary
    try:
        with open(TEST_EMAILS_PATH, 'r', encoding='utf-8') as file:
            test_emails = json.load(file)
    except FileNotFoundError:
        logging.error(f"Test emails file not found: {TEST_EMAILS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        sys.exit(1)

    # Classify emails
    results = []
    for email in test_emails:
        try:
            category = classify_email(email)
            results.append({
                "subject": email.get("subject"),
                "category": category,
                "from": email.get("from"),
                "date": email.get("date")
            })
        except Exception as e:
            logging.error(f"Failed to classify email: {email.get('subject')}. Error: {e}")
            results.append({
                "subject": email.get("subject"),
                "category": "Error",
                "from": email.get("from"),
                "date": email.get("date")
            })

    # Output results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

