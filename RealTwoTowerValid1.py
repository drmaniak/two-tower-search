import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib
import random
from utils import tokenize
from nltk.stem import PorterStemmer
import constants

random.seed(40)
punctuation_map = constants.punctuation_map

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load word-to-IDs and precomputed embeddings
word_to_ids = joblib.load("word_to_ids_v1.1.pkl")
embeds = joblib.load("embeds_v1.1.pkl")

# Hyperparameters
input_size = 300  # Embedding size
hidden_size = 256
output_size = 64
num_layers = 1
batch_size = 32
learning_rate = 0.001
num_epochs = 10
margin = 1

# Maximum sequence lengths
max_query_len = 20
max_doc_len = 50

# --- Define Custom Dataset Class ---
class TripletDataset(Dataset):
    def __init__(self, triples_file, max_query_len=40, max_doc_len=60):
        self.triples = joblib.load(triples_file)
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def text_to_ids(self, text, max_len):
        """Convert text to token IDs and embeddings."""
        tokens = tokenize(text, 
                        punctuation_map=punctuation_map, 
                        stemmer=PorterStemmer(), 
                        junk_punctuations=True)  # Simple whitespace tokenizer
        
        # Convert tokens to token IDs
        token_ids = [word_to_ids.get(token, 0) for token in tokens]  # Default to 0 if not found
        if len(token_ids) == 0:
            token_ids = [0]  # Add a dummy token (e.g., padding token)
        # Convert token IDs to embeddings (before padding)
        embeddings = [torch.from_numpy(embeds[token_id]).float() for token_id in token_ids]  # Get embeddings for each token
        
        # Now pad the sequence to max_len (padding is done after embeddings are created)
        if len(embeddings) < max_len:
            padding = [torch.zeros(input_size)] * (max_len - len(embeddings))  # Pad with zero vectors
            embeddings.extend(padding)
        else:
            embeddings = embeddings[:max_len]  # Truncate to max_len if sequence is too long
        
        # Convert to a tensor (shape: seq_len x embed_dim)
        return torch.stack(embeddings, dim=0), min(len(token_ids),max_len)  # Return embeddings and actual length

    def __getitem__(self, idx):
        sample = self.triples[idx]
        query_emb, query_len = self.text_to_ids(sample["query"], self.max_query_len)
        pos_emb, pos_len = self.text_to_ids(sample["positive"], self.max_doc_len)
        neg_emb, neg_len = self.text_to_ids(sample["negative"], self.max_doc_len)
        return query_emb, pos_emb, neg_emb, query_len, pos_len, neg_len

    def __len__(self):
        return len(self.triples)

# Create dataset and DataLoader
train_dataset = TripletDataset("train_triples_v1.1.pkl")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_dataset = TripletDataset("valid_triples_v1.1.pkl")
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- Define Model ---
class RNNTower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNTower, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hn = self.rnn(packed_x, h0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        out = self.fc(out[torch.arange(out.shape[0]), lengths - 1])  # Get last valid output
        return out

class TwoTowerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = RNNTower(input_size, hidden_size, output_size, num_layers)
        self.doc_tower = RNNTower(input_size, hidden_size, output_size, num_layers)

    def forward(self, query, query_lengths, pos_doc, pos_doc_lengths, neg_doc, neg_doc_lengths):
        query_embed = self.query_tower(query, query_lengths)
        pos_doc_embed = self.doc_tower(pos_doc, pos_doc_lengths)
        neg_doc_embed = self.doc_tower(neg_doc, neg_doc_lengths)
        return query_embed, pos_doc_embed, neg_doc_embed

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, query, pos_doc, neg_doc):
        query_norm = query / query.norm(dim=1, keepdim=True)
        pos_doc_norm = pos_doc / pos_doc.norm(dim=1, keepdim=True)
        neg_doc_norm = neg_doc / neg_doc.norm(dim=1, keepdim=True)

        pos_sim = torch.sum(query_norm * pos_doc_norm, dim=1) 
        neg_sim = torch.sum(query_norm * neg_doc_norm, dim=1)
        
        loss = torch.max(neg_sim - pos_sim + self.margin, torch.zeros_like(pos_sim)).mean()
        return loss

# Initialize model, loss function, and optimizer
model = TwoTowerModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = TripletLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
# --- Training Loop ---
print("Starting training...")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    #chunk_loss = 0
    #chunk_counter = 0

    # Training loop
    for batch_idx, (query_emb, pos_emb, neg_emb, query_len, pos_len, neg_len) in enumerate(train_loader):
        query_emb, pos_emb, neg_emb = query_emb.to(device), pos_emb.to(device), neg_emb.to(device)
        query_len, pos_len, neg_len = query_len.to(device), pos_len.to(device), neg_len.to(device)

        # Forward pass
        query_embed, pos_doc_embed, neg_doc_embed = model(query_emb, query_len,
                                                           pos_emb, pos_len,
                                                           neg_emb, neg_len)
        # Compute loss
        loss = criterion(query_embed, pos_doc_embed, neg_doc_embed)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #chunk_loss += loss.item()
        #chunk_counter += 1

        if batch_idx % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Av Chunk Loss: {loss.item():.4f}")
            #chunk_loss = 0
            #chunk_counter = 0

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()  # Set model to evaluation mode
    valid_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (query_emb, pos_emb, neg_emb, query_len, pos_len, neg_len) in enumerate(valid_loader):
            query_emb, pos_emb, neg_emb = query_emb.to(device), pos_emb.to(device), neg_emb.to(device)
            query_len, pos_len, neg_len = query_len.to(device), pos_len.to(device), neg_len.to(device)

            # Forward pass
            query_embed, pos_doc_embed, neg_doc_embed = model(query_emb, query_len,
                                                               pos_emb, pos_len,
                                                               neg_emb, neg_len)

            # Compute loss
            loss = criterion(query_embed, pos_doc_embed, neg_doc_embed)

            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_valid_loss:.4f}")

    # Optionally: Save model checkpoint based on validation loss improvement
    # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")