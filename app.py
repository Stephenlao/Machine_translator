import streamlit as st
import torch
import pickle
from translation_utils import TranslationModel, translate_sentence, en_tokenizer


def load_vocab(vocab_path):
    """Load vocabulary from a .pkl file"""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_model_and_dataset():
    checkpoint_path = "./translator_model.ckpt"
    src_vocab_path = "./src_vocab.pkl"  # Path to source vocabulary
    tgt_vocab_path = "./tgt_vocab.pkl"  # Path to target vocabulary

    # 🔹 Load vocabularies
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)

    # 🔹 Get vocab sizes
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 🔹 Determine the correct device
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load the model with vocab sizes
    model = TranslationModel.load_from_checkpoint(
        checkpoint_path, map_location=map_location, 
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size
    )

    model.eval()
    model.to(map_location)  # Move model to the correct device

    return model, src_vocab, tgt_vocab  # Return vocab as well if needed

model, src_vocab, tgt_vocab = load_model_and_dataset()



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





