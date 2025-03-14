# import os
# import streamlit as st
# import torch
# import pickle
# import gdown
# from translation_utils import TranslationModel, translate_sentence
# import lightning as pl
# import subprocess
# import spacy


# def ensure_spacy_model(model_name="en_core_web_md"):
#     try:
#         nlp = spacy.load(model_name)  # Test if model is available
#         return nlp
#     except (ImportError, OSError):
#         print(f"Installing SpaCy and downloading {model_name}...")
#         subprocess.run(["pip", "install", "spacy"], check=True)
#         subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
#         return spacy.load(model_name)

# # Load SpaCy model
# nlp_en = ensure_spacy_model("en_core_web_md")


# # Define the tokenizer for English
# def en_tokenizer(sentence):
#     return [tok.text for tok in nlp_en.tokenizer(sentence)]
# def gdrive_url(file_id):
#     return f"https://drive.google.com/uc?export=download&id={file_id}"

# def download_file(url, filename):
#     if not os.path.exists(filename):  # ✅ Check if file exists
#         gdown.download(url, filename, quiet=False)

# # Define Google Drive file IDs
# SRC_VOCAB_URL = "1K1v3I9QIXf-D6rdjufsu0_iqipF4au1q"
# TGT_VOCAB_URL = "13ePhWk0nshgUaMoAYoOyHap5iTA-YGsF" 
# MODEL_CKPT_URL = "1Owx59VZEV9Fw6GRFfZYilBjZxcpbblKd"

# src_vocab_url = gdrive_url(SRC_VOCAB_URL)
# tgt_vocab_url = gdrive_url(TGT_VOCAB_URL)
# translate_model_url = gdrive_url(MODEL_CKPT_URL)



    
# @st.cache_resource
# def load_model():
#     # ✅ Download only if files do not exist
#     download_file(src_vocab_url, "src_vocab.pkl")
#     download_file(tgt_vocab_url, "tgt_vocab.pkl")
#     download_file(translate_model_url, "translator_model.ckpt")

#     # Load vocab
#     with open("src_vocab.pkl", "rb") as f:
#         src_vocab = pickle.load(f)
#     with open("tgt_vocab.pkl", "rb") as f:
#         tgt_vocab = pickle.load(f)

#     # Load model
#     map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TranslationModel.load_from_checkpoint("translator_model.ckpt", map_location=map_location,
#                                                   src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
#     model.eval()
#     return model, src_vocab, tgt_vocab

# # Load everything
# model, src_vocab, tgt_vocab = load_model()

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


# app.py
import os
import streamlit as st
import torch
import torch.nn as nn
import pickle
import gdown
import lightning as pl
import spacy


if not spacy.util.is_package('en_core_web_sm'):
    spacy.cli.download('en_core_web_sm')
    
# Load the models
# Load SpaCy model (assumes pre-installed via requirements.txt)
nlp_en = spacy.load("en_core_web_sm")

# Define the tokenizer for English
def en_tokenizer(sentence):
    return [tok.text for tok in nlp_en.tokenizer(sentence)]

# Define PAD_IDX
PAD_IDX = 0

# Translation Model
class TranslationModel(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx=PAD_IDX, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx) if pad_idx is not None else nn.CrossEntropyLoss()

        # Initialize weights
        nn.init.xavier_uniform_(self.src_embed.weight)
        nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, src, tgt):
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_padding_mask = (src == self.pad_idx) if self.pad_idx is not None else None
        tgt_padding_mask = (tgt == self.pad_idx) if self.pad_idx is not None else None

        src_embedded = self.src_embed(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_embedded = self.tgt_embed(tgt) + self.pos_encoder[:, :tgt.size(1), :]

        output = self.transformer(
            src_embedded, tgt_embedded, 
            src_mask=src_mask, tgt_mask=tgt_mask, 
            memory_mask=None,
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc_out(output)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        output = self(src, tgt_input)
        loss = self.loss_fn(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        output = self(src, tgt_input)
        loss = self.loss_fn(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# Translation function
def translate_sentence(sentence, model, src_vocab, tgt_vocab, src_tokenizer=en_tokenizer, device='cuda', max_len=50):
    model.eval()
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    tokens = ['<sos>'] + src_tokenizer(sentence.lower()) + ['<eos>']
    src_ids = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    tgt_ids = [tgt_vocab['<sos>']]
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)

    id_to_word = {idx: word for word, idx in tgt_vocab.items()}

    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            output = model(src_tensor, tgt_tensor)
            next_token_id = output[:, -1].argmax(dim=-1).item()
            tgt_ids.append(next_token_id)
            if next_token_id == tgt_vocab['<eos>']:
                break
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            tgt_tensor = torch.cat([tgt_tensor, next_token], dim=1)

    tgt_tokens = [id_to_word.get(id, '<unk>') for id in tgt_ids]
    return ' '.join(tgt_tokens[1:-1])

# Streamlit App Logic
def gdrive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file(url, filename):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

SRC_VOCAB_URL = "1K1v3I9QIXf-D6rdjufsu0_iqipF4au1q"
TGT_VOCAB_URL = "13ePhWk0nshgUaMoAYoOyHap5iTA-YGsF"
MODEL_CKPT_URL = "1Owx59VZEV9Fw6GRFfZYilBjZxcpbblKd"

src_vocab_url = gdrive_url(SRC_VOCAB_URL)
tgt_vocab_url = gdrive_url(TGT_VOCAB_URL)
translate_model_url = gdrive_url(MODEL_CKPT_URL)

@st.cache_resource
def load_model():
    download_file(src_vocab_url, "src_vocab.pkl")
    download_file(tgt_vocab_url, "tgt_vocab.pkl")
    download_file(translate_model_url, "translator_model.ckpt")

    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TranslationModel.load_from_checkpoint(
        "translator_model.ckpt",
        map_location=map_location,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab)
    )
    model.eval()
    return model, src_vocab, tgt_vocab

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