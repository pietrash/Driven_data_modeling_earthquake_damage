import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from data.data import get_train_data, save_model


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            nn.Linear(128, 64),
            torch.nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.double()

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


def train_model():
    x, y, _ = get_train_data(encoded_x=True, encoded_y=True)
    x = x.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    input_size = len(x.columns)
    # print(f'Input size: {input_size}')
    learning_rate = 0.001

    model = NeuralNetwork(input_size).cuda()

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    x_train_tensor = torch.tensor(x_train.values).cuda()
    y_train_tensor = torch.tensor(y_train.values).cuda()
    x_test_tensor = torch.tensor(x_test.values).cuda()
    y_test_tensor = torch.tensor(y_test.values).cuda()

    num_epochs = 1000

    best_model = None
    best_loss = None

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_train_tensor)

        loss = criterion(outputs, y_train_tensor)

        # Keep the best model
        if best_model is None or best_loss > loss:
            best_model = model
            best_loss = loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every few epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model = best_model
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # test_outputs_df = pd.DataFrame(test_outputs.cpu().detach().numpy())

        # Convert tensor outputs to NumPy arrays
        y_test_numpy = y_test_tensor.cpu().numpy()
        _, predicted_indices = torch.max(test_outputs, dim=1)  # Get the indices of the maximum values
        predicted_onehot = torch.zeros_like(test_outputs)  # Initialize one-hot tensor
        predicted_onehot.scatter_(1, predicted_indices.view(-1, 1), 1)  # Set the maximum value indices to 1

        # Convert one-hot tensor to numpy array
        predicted_numpy = predicted_onehot.cpu().numpy()

        # Create a DataFrame with real and predicted values
        df = pd.DataFrame({'Real': y_test_numpy.flatten(), 'Predicted': predicted_numpy.flatten()})

        # Save the DataFrame to a CSV file
        df.to_csv('predictions.csv', index=False)

        score = f1_score(df['Real'], df['Predicted'], average='micro')

        print(f'Test score: {score:.4f}')

        save_model('NN', model, {"s": 1}, x.columns, score)


def plot_feature_importance():
    # TODO this
    return


train_model()
