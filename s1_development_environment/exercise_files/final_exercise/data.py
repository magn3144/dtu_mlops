import torch


def mnist(batch_size=256):
    """Return train and test dataloaders for MNIST."""
    train_images = []
    train_labels = []
    for i in range(6):
        train_images_path = f"c:\\Users\\magnu\\Documents\\GitHub Projects\\dtu_mlops\\data\\corruptmnist\\train_images_{i}.pt"
        train_images_file = torch.load(train_images_path)
        train_images.append(train_images_file)
        train_labels_path = f"c:\\Users\\magnu\\Documents\\GitHub Projects\\dtu_mlops\\data\\corruptmnist\\train_target_{i}.pt"
        train_labels_file = torch.load(train_labels_path)
        train_labels.append(train_labels_file)


    train_images = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_data = torch.utils.data.TensorDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_images_path = f"c:\\Users\\magnu\\Documents\\GitHub Projects\\dtu_mlops\\data\\corruptmnist\\test_images.pt"
    test_images_file = torch.load(test_images_path)
    test_labels_path = f"c:\\Users\\magnu\\Documents\\GitHub Projects\\dtu_mlops\\data\\corruptmnist\\test_target.pt"
    test_labels_file = torch.load(test_labels_path)
    test_file = torch.utils.data.TensorDataset(test_images_file, test_labels_file)
    testloader = torch.utils.data.DataLoader(test_file, batch_size=256, shuffle=True)

    return trainloader, testloader