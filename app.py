# import streamlit as st
# import torch
# import pickle
# from translation_utils import TranslationModel, translate_sentence, en_tokenizer


# def load_vocab(vocab_path):
#     """Load vocabulary from a .pkl file"""
#     with open(vocab_path, "rb") as f:
#         vocab = pickle.load(f)
#     return vocab


# def load_model_and_dataset():
#     checkpoint_path = "./translator_model.ckpt"
#     src_vocab_path = "./src_vocab.pkl"  # Path to source vocabulary
#     tgt_vocab_path = "./tgt_vocab.pkl"  # Path to target vocabulary

#     # 🔹 Load vocabularies
#     src_vocab = load_vocab(src_vocab_path)
#     tgt_vocab = load_vocab(tgt_vocab_path)

#     # 🔹 Get vocab sizes
#     src_vocab_size = len(src_vocab)
#     tgt_vocab_size = len(tgt_vocab)

#     # 🔹 Determine the correct device
#     map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 🔹 Load the model with vocab sizes
#     model = TranslationModel.load_from_checkpoint(
#         checkpoint_path, map_location=map_location, 
#         src_vocab_size=src_vocab_size, 
#         tgt_vocab_size=tgt_vocab_size
#     )

#     model.eval()
#     model.to(map_location)  # Move model to the correct device

#     return model, src_vocab, tgt_vocab  # Return vocab as well if needed

# model, src_vocab, tgt_vocab = load_model_and_dataset()



# # 🔹 Streamlit UI
# st.title("🔤 English-to-Vietnamese Translator")
# st.write("Enter an English sentence below, and the model will translate it.")

# # User input
# sentence = st.text_input("Enter your English sentence:")

# # Translate button
# if st.button("Translate"):
#     if sentence.strip():
#         translated_sentence = translate_sentence(sentence, model, src_vocab, tgt_vocab, en_tokenizer)
#         st.success(f"**Translation:** {translated_sentence}")
#     else:
#         st.warning("Please enter a sentence.")

# # Footer
# st.write("💡 Built with PyTorch & Streamlit 🚀")




import os
import streamlit as st
import torch
import pickle
import gdown
from translation_utils import TranslationModel, translate_sentence, en_tokenizer
import lightning as pl


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