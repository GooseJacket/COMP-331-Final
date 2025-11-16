import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from starter.utils import load_data, TextProcessor, convert_text_to_tensors

class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length=20):
        super(NeuralNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        # Embedding Layer
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Linear layer
        self.linear1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()

        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.gruLin = nn.Linear(hidden_size*2, output_size)  # hidden size is multiplied by 2 directions

        # Linear layer
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        emb = self.embed(x)

        lin1 = self.linear1(emb)
        relu = self.relu(lin1)

        # Pull the 3 dimensions (batch size, # of features, hidden_size) down to 2 (batch size, output_size)
        pooled = torch.mean(relu, dim=1)
        lin2 = self.linear2(pooled)
        sig = self.sig(lin2)

        return sig
        '''

        gru, _ = self.gru(x.to(dtype=torch.float32))
        gru = gru.reshape(gru.shape[0], -1)

        lin = self.gruLin(gru)

        sig = self.sig(lin)
        return sig
        '''


def train(model, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate=0.001):
    """
    Train the neural network model

    Args:
        model: The neural network model
        train_features: training features represented by token indices (tensor)
            10000 x 100
        train_labels: train labels(tensor)
            1 x 10000
        test_features: test features represented by token indices (tensor)
            5000
        test_labels: test labels (tensor)
            1x5000
        num_epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        returns are optional, you could return training history with every N epoches and losses if you want
    """

    returnHistory = ""  # the function returns history

    # Set up loss, optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), learning_rate)

    # Set up Data Loader
    trainDS = TensorDataset(train_features, train_labels)
    batch_size = 128
    trainDL = DataLoader(trainDS, batch_size=batch_size, shuffle=True)

    print("\tLearning with\t", learning_rate, "as LR and", batch_size, "in batches\n")

    # set up epochs
    for e in range(num_epochs):  # for each epoch:
        totalLoss = 0

        for i, (batchFeatures, batchLabels) in enumerate(trainDL):  # for each batch:
            # zero the gradients
            opt.zero_grad()

            # predict the batch range
            y_preds = model(batchFeatures)

            # calculate loss, step weights
            loss = loss_fn(y_preds, batchLabels)
            loss.backward()
            opt.step()
            totalLoss += loss.item()

        # keep track of epochs for history
        ep = "Epoch " + str(e + 1) + " / " + str(num_epochs) + ", loss: " + str(totalLoss / len(trainDL))
        returnHistory += ep
        # if e+1 % 10 == 0:
        print(ep)

    return returnHistory


def evaluate(model, test_features, test_labels):
    """
    Evaluate the trained model on test olddata

    Args:
        model: The trained neural network model
        test_features: (tensor)
        test_labels: (tensor)

    Returns:
        a dictionary of evaluation metrics (include test accuracy at the minimum)
        (You could import scikit-learn's metrics implementation to calculate other metrics if you want)
    """

    #######################
    # TODO: Implement the evaluation function
    # Hints:
    # 1. Use torch.no_grad() for evaluation
    # 2. Use torch.argmax() to get predicted classes
    #######################

    # for returning stats: True Positive, False Positive, False Negative, Accuracy
    TP, FP, FN, accuracy = 0, 0, 0, 0

    # set up evaluating
    model.eval()
    with torch.no_grad():

        eI = []

        # predict whole test range
        y_preds = model(test_features)
        numTests = test_labels.size()[0]

        for i in range(numTests):  # compare each prediction with true label
            pred = torch.argmax(y_preds[i])

            # add to accuracy
            accuracy += (pred == test_labels[i]).float() / numTests

            # take notes to later calculate f1
            if pred.item() == 1:  # Positive:
                if test_labels[i].item() == 1:  # True Positive
                    TP += 1
                else:  # False Positive
                    eI.append(i)
                    FP += 1
            elif test_labels[i].item() == 1:  # False Negative
                FN += 1
                eI.append(i)

        # calculate precision and recall for f1
        precision = 1.0 * TP / (TP + FP)
        recall = 1.0 * TP / (TP + FN)

    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': 2 * precision * recall / (precision + recall),
        'error_indexes': eI
    }
