import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
import string
from PIL import Image

nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit UI enhancements
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="centered")

st.title("📩 SMS Spam Detection")
st.markdown(
    """Detect if an SMS is spam or not using a machine learning model. Just paste the SMS text below and hit **Predict**!"""
)

# User input
input_sms = st.text_area(
    "Enter the SMS:", placeholder="Type or paste the SMS here...",
    key="sms_input", on_change=lambda: st.session_state.update({'result_visible': False})
)

# Initialize result visibility state
if 'result_visible' not in st.session_state:
    st.session_state['result_visible'] = False

if st.button('🚀 Predict'):
    if input_sms.strip():
        # Preprocess the SMS
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tk.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Update the result visibility
        st.session_state['result_visible'] = True
        # Display the result
        if result == 1:
            st.error("🚨 This SMS is **Spam**!")
        else:
            st.success("✅ This SMS is **Not Spam**.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Conditionally display result
# if st.session_state['result_visible']:
#     st.markdown("Result shown above") 

# Add a footer
st.markdown(
    """
    ---
    *Made with ❤️ by **Prem Javali***
    """
)
