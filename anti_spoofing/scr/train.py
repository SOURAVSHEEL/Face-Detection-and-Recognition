import torch
import os
import cv2
from deepPixBiS_model import DeepPiXBiS
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepPiXBiS().to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            heatmap, global_features = model(images)
            
            ## compute loss
            pixel_labels = labels.view(-1, 1, 1).expand_as(heatmap)
            pixel_loss = criterion(heatmap, pixel_labels)
        
            global_loss = criterion(global_features, labels.float())

            total_loss = pixel_loss + global_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


## Save the model
def save_model(model, path='deepPixBiS.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


