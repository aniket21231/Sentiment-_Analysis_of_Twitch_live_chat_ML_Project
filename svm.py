import os
import pandas as pd
import re
import nltk
import contractions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stop words and punkt tokenizer (if not already downloaded)
# nltk.download('stopwords')
# nltk.download('punkt')

# Load the CSV data into a DataFrame
csv_file = 'D:/Downloads 2/combined_data.csv'
df = pd.read_csv(csv_file)

flagged_df = df.iloc[:len(df) // 2]
non_flagged_df = df.iloc[len(df) // 2:]

# Determine the number of samples you want from each class (e.g., flagged and non-flagged)
# You can choose a balanced or imbalanced ratio based on your requirements
num_samples_per_class = 100000

# Randomly sample from each class to create a balanced dataset
flagged_sampled = flagged_df.sample(n=num_samples_per_class, random_state=50)
non_flagged_sampled = non_flagged_df.sample(n=num_samples_per_class, random_state=50)

# Concatenate the sampled DataFrames to create the balanced dataset
df = pd.concat([flagged_sampled, non_flagged_sampled], ignore_index=True)

num_rows = df.shape[0]
print(f'Number of rows before preprocessing1: {num_rows}')

# Drop rows with missing values in 'body' column
df.dropna(subset=['body'], inplace=True)

num_rows = df.shape[0]
print(f'Number of rows before preprocessing2: {num_rows}')

# Deduplicate the data based on the 'body' column
df.drop_duplicates(subset='body', inplace=True)

num_rows = df.shape[0]
print(f'Number of rows before preprocessing3: {num_rows}')

# Text Preprocessing
def preprocess_text(text):
    if pd.isna(text) or not text.strip():
        return ''  # Replace empty strings or NaN with an empty string
    
    text = text.lower()
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Normalize contractions
    text = contractions.fix(text)
    
    # Tokenize the text using NLTK's word_tokenize
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))  
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

df['body'] = df['body'].apply(preprocess_text)

# Label Encoding
df['label'] = df['label'].map({'deleted': 1, 'hidden': 1, 'nonflagged': 0})
y = df['label']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = tfidf_vectorizer.fit_transform(df['body'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (SVM)
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)

# Model Evaluation for SVM
y_pred_svm = model_svm.predict(X_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculate precision, recall, and F1-score for SVM
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Calculate classification report (includes precision, recall, and F1-score) for SVM
class_report_svm = classification_report(y_test, y_pred_svm)

# Calculate confusion matrix for SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print('Results for SVM:')
print(f'Accuracy: {accuracy_svm}')
print(f'Precision: {precision_svm}')
print(f'Recall: {recall_svm}')
print(f'F1 Score: {f1_svm}')
print('Classification Report:')
print(class_report_svm)
print('Confusion Matrix:')
print(conf_matrix_svm)
