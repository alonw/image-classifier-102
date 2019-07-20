import torch
import torchvision
from torchvision import datasets, transforms, models

def load_image_and_data(data_dir, batch_size):
    # Load the data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {}
    data_transforms['training'] = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(25),
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    data_transforms['validation'] = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    data_transforms['testing'] = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    #print('test')
    data_transforms['training']

    #TODO-BY-Alon: trasform crop only for testing and validation?
    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets['training'] = datasets.ImageFolder(train_dir, transform=data_transforms['training'])
    image_datasets['validation'] = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    image_datasets['testing'] = datasets.ImageFolder(test_dir, transform=data_transforms['testing'])


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {}
    dataloaders['training'] = torch.utils.data.DataLoader(image_datasets['training'], batch_size=32, shuffle=True)
    dataloaders['validation'] = torch.utils.data.DataLoader(image_datasets['training'], batch_size=32, shuffle=False)
    dataloaders['testing'] = torch.utils.data.DataLoader(image_datasets['training'], batch_size=32, shuffle=True)

    return dataloaders, image_datasets
