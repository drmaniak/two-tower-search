import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the Parquet file into a DataFrame
#file_path = 'sample_1k.parquet'  # Path to the Parquet file
#df = pd.read_parquet(file_path, engine='pyarrow')  # Use 'pyarrow' or 'fastparquet' as the engine


# Define the RNN Tower
class RNNTower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNTower, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, hn = self.rnn(x, h0)
        # Use the last hidden state as the output
        out = self.fc(out[:, -1, :])
        #out1 = self.fc(hn[-1])
        return out

# Define the Two-Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = RNNTower(input_size, hidden_size, output_size, num_layers)
        self.doc_tower = RNNTower(input_size, hidden_size, output_size, num_layers)

    def forward(self, query, pos_doc, neg_doc):
        # Pass inputs through their respective towers
        query_embed = self.query_tower(query)
        pos_doc_embed = self.doc_tower(pos_doc)
        neg_doc_embed = self.doc_tower(neg_doc)
        return query_embed, pos_doc_embed, neg_doc_embed

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, query, pos_doc, neg_doc):
        # Normalize embeddings to compute cosine similarity
        query_norm = query / query.norm(dim=1, keepdim=True)
        pos_doc_norm = pos_doc / pos_doc.norm(dim=1, keepdim=True)
        neg_doc_norm = neg_doc / neg_doc.norm(dim=1, keepdim=True)
        
        # Compute cosine similarities
        pos_sim = torch.sum(query_norm * pos_doc_norm, dim=1) 
        neg_sim = torch.sum(query_norm * neg_doc_norm, dim=1)
        
        # Compute triplet loss
        loss = torch.max(neg_sim - pos_sim + self.margin, torch.tensor(0)).mean()
        return loss

# Hyperparameters
input_size = 128  # Size of input embeddings
hidden_size = 256  # Hidden size of the RNN
output_size = 64  # Size of the output embeddings
num_layers = 1  # Number of RNN layers
batch_size = 32  # Batch size
sequence_length = 50  # Length of input sequences
learning_rate = 0.001
num_epochs = 10
margin = 2  # Margin for triplet loss

# Create the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = TwoTowerModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = TripletLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate dummy data for demonstration
query = torch.randn(batch_size, sequence_length, input_size).to(device)  # Query embeddings
pos_doc = torch.randn(batch_size, sequence_length, input_size).to(device)  # Positive document embeddings
neg_doc = torch.randn(batch_size, sequence_length, input_size).to(device)  # Negative document embeddings

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    query_embed, pos_doc_embed, neg_doc_embed = model(query, pos_doc, neg_doc)
    
    # Compute triplet loss
    loss = criterion(query_embed, pos_doc_embed, neg_doc_embed)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model on new data
test_query = torch.randn(1, sequence_length, input_size).to(device)
test_pos_doc = torch.randn(1, sequence_length, input_size).to(device)
test_neg_doc = torch.randn(1, sequence_length, input_size).to(device)

model.eval()
with torch.no_grad():
    # Get embeddings for the test data
    query_embed, pos_doc_embed, neg_doc_embed = model(test_query, test_pos_doc, test_neg_doc)
    
    # Compute the triplet loss for the test data
    test_loss = criterion(query_embed, pos_doc_embed, neg_doc_embed)
    
    print("Query Embedding Shape:", query_embed.shape)
    print("Positive Doc Embedding Shape:", pos_doc_embed.shape)
    print("Negative Doc Embedding Shape:", neg_doc_embed.shape)
    print(f'Test Loss: {test_loss.item():.4f}')
