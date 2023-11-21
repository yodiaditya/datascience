import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_size, compression_size):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(nn.Linear(input_size, compression_size),
                                     nn.ReLU(True))
        # decoder
        self.decoder = nn.Sequential(nn.Linear(compression_size, input_size),
                                     nn.ReLU(True))
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_size = 1000
compression_size = 100

model = AutoEncoder(input_size, compression_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = torch.randn(100, input_size)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()

    if epochs % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


compressed_data = model.encoder(data)
reconstructed_data = model.decoder(compressed_data)

