"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

parser = argparse.ArgumentParser(description='Image Classification Data Loader')

# Add arguments
parser.add_argument('--train_dir', type=str, required=True,
                    help='Path to the training data directory')
parser.add_argument('--test_dir', type=str, required=True,
                    help='Path to the testing data directory')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Mini-batch size (default: 32)')
parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                    help='Number of worker threads for data loading (default: number of CPU cores)')
parser.add_argument('--n_epochs', type=int, default=5,
                    help='Number of epochs (default: 5)')
parser.add_argument('--h_units', type=int, default=16,
                    help='Number of hidden units in hidden layer (default: 16)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--model_name', type=str, help='Model name', required=True)

# Parse the arguments
args = parser.parse_args()

# Setup hyperparameters
n_epochs = args.n_epochs
batch_size = args.batch_size
hid_units = args.h_units
lr = args.lr
n_workers = args.num_workers
model_name = args.model_name

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, \
class_names = data_setup.create_dataset_dataloader(train_dir=train_dir,
                                            test_dir=test_dir,
                                            transform=data_transform,
                                            batch_size=batch_size, n_workers=n_workers)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(in_channels=3, hid_units=hid_units,
                              out_classes=len(class_names), image_shape=64).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr)

# training and testing with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=n_epochs,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=model_name)
