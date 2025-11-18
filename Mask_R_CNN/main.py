from engine import train_one_epoch, evaluate
import utils
import torch
import transforms as T
from loader import PennFudanDataset,PennFudanDataset_train
from model import get_model_instance_segmentation
import os
from settings import MODEL_NAME,EPOCH,BATCH_SIZE,NUM_CLASS,IMAGE

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = NUM_CLASS
    # use our dataset and defined transformations
    dataset = PennFudanDataset_train(IMAGE, get_transform(train=True))
    dataset_test = PennFudanDataset_train(IMAGE, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])
    print(len(dataset), len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = EPOCH

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    #保存训练好的模型
    path='./Weights'
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(model.state_dict(),os.path.join(path,MODEL_NAME))

    print("That's it!")

if __name__=='__main__':
    main()