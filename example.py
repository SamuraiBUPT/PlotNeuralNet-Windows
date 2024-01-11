import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a simple 3-layer convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc(x)
        return x

# Initialize the CNN
model = SimpleCNN()

# Set up hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(num_epochs, train_loader, model, criterion, optimizer, device):
    print("Training started")
    model.train()
    model.to(device)
    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training complete")
    # save the model
    torch.save(model.state_dict(), './model_ckpt/model.pth')
    
    
def eval(model, test_loader, device):
    print("Evaluation started")
    model.eval()
    model.to(device)
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    print("Evaluation complete")
    
from neuplotlib import TorchPlot
    
def test_infer(model, test_loader, device):
    model.eval()
    model.load_state_dict(torch.load('./model_ckpt/model.pth'))
    model.to(device)
    # Test the model
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # test
            tp = TorchPlot(config={})
            tp.analyze_net(model, images)
            
            # outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            break

if __name__ == '__main__':
    # test_infer(model, test_loader, device)
    from neuplotlib import *
    arch2 = [
        to_head(),
        to_color(),
        to_begin(),

        to_Conv(name="conv1", z_label=64, x_label=3, 
                base="(0,0,0)", offset="(0,0,0)", 
                height=64, depth=64, width=1, 
                caption="Conv1"),
        to_Pool("pool1", base="(conv1-east)", offset="(0,0,0)",
                height=48, depth=48, width=1),
        
        to_Conv(name="conv2", z_label=128, x_label=64, base="(pool1-east)", offset="(4,0,0)", 
                height=48, depth=48, width=3, caption="Conv2"),
        to_Pool("pool2", base="(conv2-east)", offset="(0,0,0)", 
                height=30, depth=30, width=3, caption="MaxPooling"),
        
        to_connection("pool1", "conv2"),

        to_Conv(name="conv3", z_label=1, x_label=128 * 8 * 8 * 8, 
                base="(pool2-east)", offset="(4,0,0)",
                height=2, depth=2, width=10, caption="Flatten"),
        to_connection("pool2", "conv3"),

        # fc1
        to_SoftMax(name='fc1', z_label=128 * 8 * 8 * 8, 
                   base="(conv3-east)", offset="(4,0,0)",  
                   width=1.5, height=1.5, depth=100, opacity=0.8, caption='FC1'),
        to_connection("conv3", "fc1"),
        
        # fc2
        to_SoftMax(name='fc2', z_label=512, base="(fc1-east)", offset="(1.5,0,0)", 
                   width=1.5, height=1.5, depth=100, opacity=0.8, caption='FC2'),
        to_connection("fc1", "fc2"),
        # fc1
        to_SoftMax(name='fc3', z_label=256, base="(fc2-east)", offset="(1.5,0,0)",  
                   width=1.5, height=1.5, depth=70, opacity=0.8, caption='FC3'),
        to_connection("fc2", "fc3"),
        to_end()
    ]
    filename = 'test.tex'
    
    model.eval()
    model.load_state_dict(torch.load('./model_ckpt/model.pth'))
    model.to(device)
    # Test the model
    
    tp = TorchPlot(config={})
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # test
            tp.analyze_net(model, images)
            break
    tp.generate(arch2, filename)