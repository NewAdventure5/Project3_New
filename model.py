import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000 , dropout: float = 0.7) -> None:
# previous num_classes 1000
        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        
        #self.conv_layers = nn.Sequential(
        #    nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(64),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(128),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Dropout(dropout),
        #    nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Dropout(dropout),
        #    nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Dropout(dropout)
        #)
        
        #self.linear_layers = nn.Sequential(
        #    nn.Linear(in_features=512*7*7, out_features=256),
        #    nn.ReLU(),
        #   
        #    nn.Dropout(dropout),
        #    nn.Linear(in_features=256, out_features=num_classes)
        #)
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,  stride = 1, padding=1) 
        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
        
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
       
        
            
        
            
            
            
            
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        #x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        #x = x.view(x.size(0),-1)
        #x = self.linear_layers(x)
        
        x = self.pool2((F.relu(self.conv1(x))))
        print(f'Output after conv1 and pool: {x.shape}')
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        print(f'Output after conv2 and pool: {x.shape}')
        x = self.pool2(self.bn3(F.relu(self.conv3(x))))
        print(f'Output after conv3 and pool: {x.shape}')
        x = self.pool2(self.bn4(F.relu(self.conv4(x))))
        print(f'Output after conv4 and pool: {x.shape}')
        x = self.pool2(self.bn5(F.relu(self.conv5(x))))
        print(f'Output after conv5 and pool: {x.shape}')
        
        
        
        x = self.flatten(x)
       
        print(f'Output after flattening: {x.shape}')
        
        x = self.dropout(F.relu(self.fc1(x)))
        print(f'Output after fc1: {x.shape}')
        x = self.dropout(F.relu(self.fc2(x)))  
        
        
        
        print(f'Output after fc2: {x.shape}')
        x = self.fc3(x)
        print(f'Output after fc3: {x.shape}')
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
