import click
import torch
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import numpy as np

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


def training_curve_plot(train_losses, test_accuracies, evaluate_every):
    """Plot the training curve, and save the plot to a file."""
    train_losses = np.array(train_losses) / np.max(train_losses)
    plt.plot(np.arange(len(train_losses)) * evaluate_every, train_losses, label='Training loss normalized')
    plt.plot(np.arange(len(test_accuracies)) * evaluate_every, test_accuracies, label='Test accuracy')
    plt.legend()
    plt.savefig('training_curve.png')


def evaluate_(model_checkpoint):
    """Evaluate a trained model."""

    from_file = False
    if type(model_checkpoint) == str:
        model = torch.load(model_checkpoint)
        from_file = True
    else:
        model = model_checkpoint

    _, test_set = mnist()
    all_equals = []
    for images, labels in test_set:
        images = images.unsqueeze(1)
        output = model(images)
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        all_equals.append(equals)

    all_equals = torch.cat(all_equals)
    accuracy = torch.mean(all_equals.type(torch.FloatTensor))
    if from_file:
        print(f'Accuracy: {accuracy.item()*100}%')
    return accuracy.item()


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--evaluate_every", default=2000, help="the amount of steps between evaluations")
@click.option("--epochs", default=1, help="the amount of epochs to train for")
def train(lr, batch_size, evaluate_every, epochs):
    """Train a model on MNIST."""

    model = MyAwesomeModel()
    train_set, _ = mnist(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []

    data_points_trained = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images = images.unsqueeze(1)
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            data_points_trained += len(images)
            if len(train_losses) < data_points_trained // evaluate_every:
                train_losses.append(loss.item())
                accuracy = evaluate_(model)
                test_accuracies.append(accuracy)
        else:
            print(f"Training loss: {running_loss/len(train_set)}")

        torch.save(model, f"model_{e}.pt")

    training_curve_plot(train_losses, test_accuracies, evaluate_every)  


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    evaluate_(model_checkpoint)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
