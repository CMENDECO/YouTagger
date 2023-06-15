import re

def clean_text(text):
    if isinstance(text, str):
        # Remove special characters, emails, and links using regex
        text = re.sub(r'http\S+', '', text)  # Remove links
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text
    else:
        return ''

def preprocess_data(data):
    # Join Title, Description, and Tags columns
    data['Text'] = data['Title'] + ' ' + data['Description'] + ' ' + data['Tags']

    # Clean the text using the clean_text function
    data['Cleaned_Text'] = data['Text'].apply(clean_text)

    # Transform the cleaned text using the TF-IDF vectorizer
    preprocessed_data = tfidf_vectorizer.transform(data['Cleaned_Text'])

    return preprocessed_data
