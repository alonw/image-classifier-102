import torch
import torchvision
from torchvision import datasets, transforms, models

def load_image_and_data(data_dir="flower_data", batch_size=32):
    # Load the data
    # data_dir = 'flowers'
    print("load_image_and_data, data_dir is " + data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    # data_transforms = {}
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(25),
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    #print('test')
    #data_transforms['training']

    #TODO-BY-Alon: trasform crop only for testing and validation?
    # TODO: Load the datasets with ImageFolder
    # image_datasets = {}
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_dataset = datasets.ImageFolder(valid_dir, transform=validate_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    # print('############ about to print dataloader[training] #######')
    # print(dataloaders['training'])

    return train_dataloader, validate_dataloader, test_dataset
