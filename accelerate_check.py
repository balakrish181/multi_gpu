import torch
from torchvision import transforms, datasets, models
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

# Initialize the Accelerator
accelerate = Accelerator()

# Define transformations and dataset
weights = models.ViT_L_32_Weights.DEFAULT
transform = weights.transforms

# Load CIFAR-10 dataset
train_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Load the pre-trained ViT model and modify the classifier head
model = models.vit_l_32(weights=weights)
model.head = torch.nn.Linear(1024, 10)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Prepare everything with Accelerator
model, optimizer, train_loader, test_loader = accelerate.prepare(model, optimizer, train_loader, test_loader)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        logits = model(X)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        accelerate.backward(loss)
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}/{num_epochs}"):
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the trained model
accelerate.wait_for_everyone()
unwrapped_model = accelerate.unwrap_model(model)
torch.save(unwrapped_model.state_dict(), "vit_cifar10.pth")
