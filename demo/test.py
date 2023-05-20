import streamlit as st
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import tensorflow_text
import pythainlp
import re
import ast
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit import components


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# def load_data():
#     trained_model = tf.keras.models.load_model('./Users/pornchanan/Desktop/getproject/transformer_USE_without_texttoken')
#     return trained_model

# def train_model(df):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(df["text_token"])
#     y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
#     y = np.array(y)
#     model = RandomForestClassifier()
#     model.fit(X, y)
#     return vectorizer, model

# def predict(vectorizer, model, text):
#     X = vectorizer.transform([text])
#     prediction = model.predict(X)
#     return prediction


def cln(rawtext):
  stopwords = pythainlp.corpus.common.thai_stopwords()
  symbols_to_remove = "!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~…꒰꒱“”◤◥•~"
  x = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)|(\n|\r|\s|\t|\d+)|(_|ๆ|ฯ|฿)|([\u0E4D-\u0E7F]+)","",str(rawtext)).split())
  x = re.sub(f'[{re.escape(symbols_to_remove)}]', '', x)
  x = " ".join([i for i in x.lower().split(" ") if not i.isdigit()])
  x = x.replace("เเ", "แ").replace("\u200b","")
  RE_EMOJI = re.compile(u'([\U00010000-\U0010ffff]|[\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])|([\U0001F600-\U0001F64F])|([\U0001F97A]|([\u2B55]))')
  x = RE_EMOJI.sub(r'',x)
  x = re.sub(r'([\u0E01-\u0E5B])\1+$', r'\1', x)
  x = [word for word in x if word not in stopwords]
  return "".join(x)

def predict(text):
    trained_model = tf.keras.models.load_model("/Users/pornchanan/Desktop/getproject/transformer_USE_without_texttoken", compile=False)
    prediction = trained_model.predict([text]) 
    return prediction


# def transform_label(prediction):
#     label_names = ['positive', 'negative']
#     predicted_labels = [label_names[i] for i in range(len(prediction)) if prediction[i] == 1]
#     return predicted_labels

def main():
    st.title("Text Classification App")
    
   
    # df = load_data()
    # vectorizer, model = train_model(df)
    col1, col2 = st.columns(2)
    with col1:
        rawtext = st.text_input("Enter text to classify") 
    with col2:
        threshold = st.slider("Level of threshold", 0.1, 0.9, 0.4, 0.1)
    if st.button("Classify"):
        text = cln(rawtext)
        prediction = predict(text)
        prediction = np.array(prediction)
        # prediction = np.array2string(prediction, separator=', ')
        # prediction = ast.literal_eval(prediction)
        a = prediction > threshold
        prediction_neg_pos = np.multiply(a, 1)
        # Print the result
        print(threshold)
        print(text)
        print(prediction[0])
        dictt = {0:'negative',1:'positive'}
        # lable = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        prediction_neg_pos = [dictt[i] for i in prediction_neg_pos[0]]
        print(prediction_neg_pos)
        df = pd.DataFrame({"toxic": [prediction_neg_pos[0]], 
        "severe_toxic": [prediction_neg_pos[1]],
        "obscene": [prediction_neg_pos[2]],
        "threat": [prediction_neg_pos[3]],
        "insult": [prediction_neg_pos[4]],
        "identity_hate": [prediction_neg_pos[5]]})
        df = df.rename(index={0: "result"})
        st.table(df)
        fig = px.bar(
            x= [round(i*100, 2) for i in prediction[0]] ,
            y=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
            orientation='h',
            text = [str(round(i*100, 2))+' %' for i in prediction[0]])
        # Update layout to show text on bars
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.write(fig)
if __name__ == '__main__':
    main()