import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from collections import OrderedDict

def create_model(arch="vgg19", hidden_units=4096, learning_rate=0.01):
    '''
        Create the model using the desired archtecture and layers. 

        Inputs:
        arch - thid could be VGG19 (default). 
        This would be the right place to implement additional options. 

        Outputs:
        A loaded model, created according to the supplied checkpoint. Not trained.
    '''
    if(arch.lower() == "vgg19"):
        print("Implementing VGG19")
        model = models.vgg19(pretrained=True)
        input_features = 25088
    else:
        print("Sorry, only VGG19 implemented. You are welcome to contribute your desired model!")
        return 0

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    
    # epochs = 1
    # learning_rate = 0.001
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #model.load_state_dict(checkpoint['state_dict'])
    
    print("Model suceesfuly created, yey!")
    return model, criterion, optimizer

def save_model(model, datasets, learning_rate, epochs, criterion, optimizer, hidden_units, arch="vgg19", batch_size=32):
    '''
        Saves the model to the checkpoint.pth file (path supplied or deafault)

        Inputs:
        The model itsef, the datasets used, learning rate and epochs to run, 
        the loss function, optimizer, hidden units to be implemented,  arch (only vgg19 supprted currently)

        Ouput:
        Saving to checkpoint file, no other output
    '''
    #def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
    #def save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):

    if(arch.lower() == "vgg19"):
        print("Saving model with vgg19 archtecture, using 25088 input features..")
        input_features = 25088
    else:
        print("Sorry, only vgg19 supprted currently")
        return 0

    checkpoint = {'input_size': input_features,
                'output_size': 102,
                'hidden_units': hidden_units,
                'arch': arch,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion, #loss function
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint,  'checkpoint.pth')
    print("Model successfuly saved")
    #return ''

def load_model(checkpoint_path='checkpoint.pth'):
    '''
        Loads a model using the checkpoint

        Inputs: checkpoint file
        Output: The model 
    '''

    print("Loading model...")
    checkpoint = torch.load(checkpoint_path)

    if checkpoint['arch'].lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'].lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'].lower() == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Sorry, {} model not supported?".format(checkpoint['arch']))
        return 0
        
    return model


def train_model(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):
    model.train() # Puts model into training mode
    print_every = 20
    steps = 0
    use_gpu = True
    
    # Check to see whether GPU is available
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()
    else:
        model.cpu()
    
    # Iterates through each training pass based on #epochs & GPU/CPU
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

            # Forward and backward passes
            optimizer.zero_grad() # zero's out the gradient, otherwise will keep adding
            output = model.forward(inputs) # Forward propogation
            loss = criterion(output, labels) # Calculates loss
            loss.backward() # Calculates gradient
            optimizer.step() # Updates weights based on gradient & learning rate
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validation_loader)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(accuracy))




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def sanity_check():

    return ''
