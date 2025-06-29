import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, device, num_epochs=10):
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            heatmap, global_features = model(images)

            # Resize labels to match heatmap
            pixel_labels = labels.view(-1, 1, 1).expand_as(heatmap)
            pixel_loss = criterion(heatmap, pixel_labels)

            global_loss = criterion(global_features, labels.float())

            total_loss = pixel_loss + global_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



