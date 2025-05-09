import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the saved Random Forest model and vectorizer
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """Function to clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("📩 Spam or Ham Message Classifier")
st.write("🔍 Enter a message below to check if it's spam or not!")


# User Input with an Example Placeholder
user_input = st.text_area("✍️ Type your message here:", placeholder="Example: Congratulations! You've won a free gift. Click here to claim now!")

# Prediction Button
if st.button("🚀 Check Message"):
    if user_input.strip():
        with st.spinner("Analyzing message..."):
            processed_text = preprocess_text(user_input)
            transformed_text = vectorizer.transform([processed_text])
            prediction = model.predict(transformed_text)[0]
            result = "🚨 **Spam**" if prediction == 1 else "✅ **Not Spam**"
            st.success(f"Prediction: {result}")
            
            # Extra Info
            if prediction == 1:
                st.warning("Be cautious! This message might be a scam or promotional spam.")
            else:
                st.info("This message appears to be legitimate.")
    else:
        st.error("⚠️ Please enter a message to classify.")

st.markdown("---")

