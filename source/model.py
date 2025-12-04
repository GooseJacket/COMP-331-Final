import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        num_layers = 1
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.gruLin = nn.Linear(hidden_size*2, output_size)  # hidden size is multiplied by 2 directions

        # Linear layer
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        '''
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
        return lin



def train(model, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate=.00001):
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
    """

    returnHistory = ""  # the function returns history

    # Set up loss, optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), learning_rate)

    # Set up Data Loader
    trainDS = TensorDataset(train_features, train_labels)
    batch_size = 516
    trainDL = DataLoader(trainDS, batch_size=batch_size, shuffle=True)

    print("\tLearning with\t", learning_rate, "as LR and", batch_size, "in batches\n")

    print("\t" + "-" * int(len(train_features)/batch_size), end="")
    if len(train_features) % batch_size > 0:
        print("-")
    else:
        print()

    # set up epochs
    for e in range(num_epochs):  # for each epoch:
        totalLoss = 0

        print("\t", end="")

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
            print("~", end="")

        # keep track of epochs for history
        ep = "Epoch " + str(e + 1) + " / " + str(num_epochs) + ", loss: " + str(totalLoss / len(trainDL))
        returnHistory += ep
        # if e+1 % 10 == 0:
        print("\n", ep)

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


        # predict whole test range
        y_preds = model(test_features)
        numTests = test_labels.size()[0]

        soft_max = nn.Softmax(dim=None)

        num_features = len(y_preds[0])
        guesses = [[a for a in range(num_features + 1)] for b in range(num_features + 1)]

        for i in range(numTests):  # compare each prediction with true label
            pred = y_preds[i]
            pred = soft_max(pred)
            pred = torch.argmax(pred)

            # add to accuracy
            accuracy += (pred == test_labels[i]).float() / numTests

            # row = correct, column = prediction
            guesses[test_labels[i]][pred.item()] += 1
            guesses[test_labels[i]][-1] += 1
            guesses[-1][pred.item()] += 1
            guesses[-1][-1] += 1

            """
                A   B   C   T
            A   +=1         +=1
            B
            C
            T   +=1         +=1
            """


        # calculate precision and recall for f1
        #precision = 1.0 * TP / (TP + FP)
        #recall = 1.0 * TP / (TP + FN)

    return {
        'test_accuracy': accuracy,
        #'test_precision': precision,
        #'test_recall': recall,
        'test_f1': 0.0,  # 2 * precision * recall / (precision + recall),
        'guesses': [[str(g).rjust(4) for g in r] for r in guesses]
    }