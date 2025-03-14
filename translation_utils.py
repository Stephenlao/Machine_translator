# translation_utils.py
import torch
import torch.nn as nn
import lightning as pl
import spacy
import subprocess

# Ensure SpaCy model is installed
def ensure_spacy_model(model_name="en_core_web_md"):
    try:
        nlp = spacy.load(model_name)  # Test if model is available
        return nlp
    except (ImportError, OSError):
        print(f"Installing SpaCy and downloading {model_name}...")
        subprocess.run(["pip", "install", "spacy==3.7.2"], check=True)
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

# Load SpaCy model
nlp_en = ensure_spacy_model("en_core_web_md")

# Define the tokenizer for English
def en_tokenizer(sentence):
    return [tok.text for tok in nlp_en.tokenizer(sentence)]


# 1
PAD_IDX = 0



# 2
class TranslationModel(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx=None, d_model=512, nhead=8, 
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
     
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))  # Max seq len = 5000
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx) if pad_idx is not None else nn.CrossEntropyLoss()

        # Initialize weights
        nn.init.xavier_uniform_(self.src_embed.weight)
        nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(self, src, tgt):
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)  # Lower initial LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def translate_sentence(sentence, model, src_vocab, tgt_vocab, src_tokenizer, device='cuda', max_len=50):
    model.eval()  # Set the model to evaluation mode

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Tokenize and add <sos> and <eos>
    tokens = ['<sos>'] + src_tokenizer(sentence.lower()) + ['<eos>']
    src_ids = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]

    # Convert to tensor and move to device
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Initialize target sequence with <sos>
    tgt_ids = [tgt_vocab['<sos>']]
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)  # (1, 1)

    # Invert tgt_vocab for ID-to-token mapping
    id_to_word = {idx: word for word, idx in tgt_vocab.items()}

    # Inference loop
    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            output = model(src_tensor, tgt_tensor)
            
            next_token_id = output[:, -1].argmax(dim=-1).item()
            tgt_ids.append(next_token_id)
            
            if next_token_id == tgt_vocab['<eos>']:
                break
            
            # Append next token correctly
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            tgt_tensor = torch.cat([tgt_tensor, next_token], dim=1)

    # Convert IDs to tokens
    tgt_tokens = [id_to_word.get(id, '<unk>') for id in tgt_ids]

    # Remove <sos> and <eos>
    translated_sentence = ' '.join(tgt_tokens[1:-1])

    return translated_sentence
