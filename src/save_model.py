import torch
from tft_train import model  # 👈 if model is still in memory

# Save the trained model weights
torch.save(model.state_dict(), "tft_model.pt")

print(" Model saved as 'tft_model.pt'")
