import tkinter as tk
from tkinter import messagebox
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load NLTK stopwords and lemmatizer
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load CountVectorizer and Classifier from pickle files
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    return ' '.join(review)

# Function to perform classification
def classify_email():
    input_text = text_entry.get("1.0",'end-1c')
    if input_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter an email text.")
        return
    
    processed_text = preprocess_text(input_text)
    test_vec = cv.transform([processed_text])
    prediction = classifier.predict(test_vec)[0]
    
    if prediction == 1:
        result_label.config(text="Spam Mail", fg="red")
    else:
        result_label.config(text="Not Spam Mail", fg="green")

# GUI setup
root = tk.Tk()
root.title("Spam Classifier")

# Text Entry
text_label = tk.Label(root, text="Enter Email Text:")
text_label.pack(pady=10)

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict", command=classify_email)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", fg="black", font=("Helvetica", 16))
result_label.pack()

root.mainloop()
