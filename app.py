import os
import streamlit as st
import torch
import pickle
import gdown
from translation_utils import TranslationModel, translate_sentence, en_tokenizer
import lightning as pl
import subprocess
import spacy


def ensure_spacy_model(model_name="en_core_web_md"):
    try:
        nlp = spacy.load(model_name)  # Test if model is available
        return nlp
    except (ImportError, OSError):
        print(f"Installing SpaCy and downloading {model_name}...")
        subprocess.run(["pip", "install", "spacy"], check=True)
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

# Load SpaCy model
nlp_en = ensure_spacy_model("en_core_web_md")


# Define the tokenizer for English
def en_tokenizer(sentence):
    return [tok.text for tok in nlp_en.tokenizer(sentence)]
def gdrive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file(url, filename):
    if not os.path.exists(filename):  # ✅ Check if file exists
        gdown.download(url, filename, quiet=False)

# Define Google Drive file IDs
SRC_VOCAB_URL = "1K1v3I9QIXf-D6rdjufsu0_iqipF4au1q"
TGT_VOCAB_URL = "13ePhWk0nshgUaMoAYoOyHap5iTA-YGsF" 
MODEL_CKPT_URL = "1Owx59VZEV9Fw6GRFfZYilBjZxcpbblKd"

src_vocab_url = gdrive_url(SRC_VOCAB_URL)
tgt_vocab_url = gdrive_url(TGT_VOCAB_URL)
translate_model_url = gdrive_url(MODEL_CKPT_URL)



    
@st.cache_resource
def load_model():
    # ✅ Download only if files do not exist
    download_file(src_vocab_url, "src_vocab.pkl")
    download_file(tgt_vocab_url, "tgt_vocab.pkl")
    download_file(translate_model_url, "translator_model.ckpt")

    # Load vocab
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    # Load model
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TranslationModel.load_from_checkpoint("translator_model.ckpt", map_location=map_location,
                                                  src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
    model.eval()
    return model, src_vocab, tgt_vocab

# Load everything
model, src_vocab, tgt_vocab = load_model()

# 🔹 Streamlit UI
st.title("🔤 English-to-Vietnamese Translator")
st.write("Enter an English sentence below, and the model will translate it.")

# User input
sentence = st.text_input("Enter your English sentence:")

# Translate button
if st.button("Translate"):
    if sentence.strip():
        translated_sentence = translate_sentence(sentence, model, src_vocab, tgt_vocab, en_tokenizer)
        st.success(f"**Translation:** {translated_sentence}")
    else:
        st.warning("Please enter a sentence.")

# Footer
st.write("💡 Built with PyTorch & Streamlit 🚀")