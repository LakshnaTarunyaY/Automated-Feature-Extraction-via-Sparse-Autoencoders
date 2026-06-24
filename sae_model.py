
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self,x):
      return self.relu(self.encoder(x))

    def decode(self,z):
      return self.decoder(z)

    def forward(self, x):
        z = self.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z
