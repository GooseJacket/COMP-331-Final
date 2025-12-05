import torch
from model import train, evaluate, NeuralNetwork
from utils import load_data, TextProcessor, convert_text_to_tensors

if __name__ == "__main__":

    ####################
    # TODO: If applicable, modify anything below this line
    # according to your model configuration
    # and to suit your need (naming changes, parameter changes,
    # additional statements and/or functions)
    ####################

    # Load training and test data
    train_texts, train_labels = load_data("data/trainSansSh.txt")
    test_texts, test_labels = load_data("data/testSansSh.txt")

    # DEBUGGING DATA (smaller to speed up training and ensure no dumb mistakes)
   # train_texts, train_labels = train_texts[0:10000], train_labels[0:1000]
    # test_texts, test_labels = test_texts[0:1000], test_labels[0:1000]

    # Preprocess text
    processor = TextProcessor(vocab_size=20000)
    processor.build_vocab(train_texts)

    # Convert text documents to tensor representations of word indices
    max_length = 500
    train_features = convert_text_to_tensors(train_texts, processor, max_length)
    test_features = convert_text_to_tensors(test_texts, processor, max_length)

    # Create a neural network model
    # Modify the hyperparameters according to your model architecture
    vocab_size = len(processor.word_to_idx)
    embedding_dim = max_length
    hidden_size = 256
    output_size = 4

    print("\nVariables:")
    print("\tTraining size\t", len(train_texts), "texts x", max_length, "tokens")
    print("\tTest size\t\t", len(test_texts), "texts x", max_length, "tokens")
    print("\tForward pass\t", embedding_dim, "embed ->", hidden_size, "hidden ->", output_size, "classes")

    model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)

    # Set up the run
    num_epochs = 0
    outfile = 'trained_model.pth'
    if input("Load Model? 0|1\n") == "1":
        model.load_state_dict(torch.load(outfile))
        print("Model loaded.")
    else:
        print("Model reset.")
    try:
        num_epochs = int(input("How many epochs?\n"))
    except:
        num_epochs = 25
    finally:
        if num_epochs == 0:
            num_epochs = 1
        print("Okay, ", num_epochs, "Epochs.")

    # Train
    training_history = train(model, train_features, train_labels, test_features, test_labels, num_epochs)

    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)

    print()
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    # print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")

    print("\nGuesses in Each Category: (real average =", len(test_texts)/output_size, ")")
    print("Guess=", "\t".join([str(i).rjust(4) for i in range(len(evaluation_results['guesses'])-1)]), "Total", sep="\t")
    for i in range(len(evaluation_results['guesses'])):
        print("Real=" + str(i), "\t".join(evaluation_results['guesses'][i]), sep="\t")

    # Save model weights to file
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")