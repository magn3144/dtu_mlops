import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print("lr:", lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 5
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
        else:
            print(f"Training loss: {running_loss/len(train_set)}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    output = model(test_set)
    top_p, top_class = output.topk(1, dim=1)
    print(top_class.shape)
    print(top_class[:10,:])
    images, labels = next(iter(test_set))
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')
    return accuracy.item()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
