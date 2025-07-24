

import os
import streamlit as st
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
# Hide OpenAI API Key:

st.title('RTPGPT')
st.subheader('Archie - Snowbird Smart Assistant')

@st.cache_resource(show_spinner='')
def load_model():
  # All Relevant Model code: 

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

  return model

# Caching model so that the app runs faster:
model = load_model()

@st.cache_data(show_spinner="Loading and embedding docs...")
def load_data():
  # Open RTP Docs: 

  with open("RTP_Website_Docs_Plaintext.txt", "r", encoding="utf-8") as f:
      raw_text = f.read()

  # Aspenware docs:

  with open("aspenware.txt", "r", encoding="utf-8") as f:
      raw_text_2 = f.read()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 100
  )

  my_chunks = text_splitter.create_documents([raw_text,raw_text_2])


  docs = [doc.page_content for doc in my_chunks]
  doc_embeddings = model.encode(docs, show_progress_bar=True)

  return my_chunks, doc_embeddings

my_chunks, doc_embeddings = load_data()

# OpenAI Client:

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Prompter:

def ask_rtp_question(question, your_chunks, doc_embedding,top_k=3, last_message = ""):
  # Debug:
  try:
    question_embedding = model.encode(question)
    doc_embedding = np.array(doc_embedding)
    similarities = cosine_similarity(question_embedding.reshape(1,-1), doc_embedding)[0]
    # # Debug:
    # st.markdown(f"**[Debug]** Question embedding shape: '{getattr(question_embedding, 'shape', 'no shape')}'")
    # st.markdown(f"**[Debug]** Question embedding type: '{type(question_embedding)}'")
    # st.markdown(f"**[Debug]** Doc embedding shape: '{getattr(doc_embedding, 'shape', 'no shape')}'")
    # st.markdown(f"**[Debug]** Doc embedding type: '{type(doc_embedding)}'")

    try:
      top_indices = np.argsort(similarities)[-top_k:][::-1]
      top_chunks = [your_chunks[int(i)].page_content for i in top_indices]

      # st.markdown(f"**[Debug]** top_indices: '{top_indices}'")
    except:
      st.markdown('Cannot load top indices')
    
    context = "\n\n".join(top_chunks)
    
    # Session memory retention logic:
    if len(last_message) > 1: 
      context = context + last_message['answer']
    else:
      st.markdown('First Answer:')

    prompt = f"""You are a helpful and sociable/friendly assistant trained on Activeware's RTP documentation for ski resorts and Aspenware. 
    Nothing from the eStore documentation is to be brought up. Nothing related to bStore is to be brought up.
    Remember that new ticket types are organized under product headers.
    Based on the following context, answer
    the question as clearly as possible. 
    Context: {context}
    Question: {question}
    Answer:"""

    response = client.chat.completions.create(
      model="gpt-4.1",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2
    )
    return response.choices[0].message.content
  except Exception as e:
    st.markdown('**[Debug]** Embedding Error:')
    st.markdown(str(e))
    return 'Sorry, something went wrong while embedding your question.'

# Actual prompting UI / asked question information goes here:

if "history" not in st.session_state:
   st.session_state.history = []

question = st.text_input("Ask me a question about ski resort systems!")



# Clear history button:

col1, col2, col3 = st.columns([5,10,5])
with col3:
  clear_history = st.button("Clear Chat History")
with col1:
  show_history = st.button("Show History")


if clear_history:
  st.session_state.history.clear()

if question:

    with st.spinner("Thinking..."):

      if len(st.session_state.history) !=0:
        result = ask_rtp_question(question, my_chunks, doc_embeddings, last_message=st.session_state.history[0])
      else:
        result = ask_rtp_question(question, my_chunks, doc_embeddings)
      
      st.session_state.history.append({"question": question, "answer": result})
      
      st.markdown("### Answer:")
      st.write(result)
      
if show_history:
  if len(st.session_state.history) > 1:
    if len(st.session_state.history) == 2:
      st.markdown("### Conversation History")
    for qa in st.session_state.history:
      st.markdown("--------------------------")
      st.markdown(f"**You:** {qa['question']}")
      st.markdown(f"**Assistant:** {qa['answer']}")
      