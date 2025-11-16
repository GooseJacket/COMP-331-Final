import torch
from oldsource.model import train, evaluate, NeuralNetwork
from oldsource.utils import load_data, TextProcessor, convert_text_to_tensors

if __name__ == "__main__":

    ####################
    # TODO: If applicable, modify anything below this line
    # according to your model configuration
    # and to suit your need (naming changes, parameter changes,
    # additional statements and/or functions)
    ####################

    # Load training and test olddata
    train_texts, train_labels = load_data("../olddata/total_train.txt")
    test_texts, test_labels = load_data("../olddata/total_test.txt")

    # Preprocess text
    processor = TextProcessor(vocab_size=10000)
    processor.build_vocab(train_texts)

    # Convert text documents to tensor representations of word indices
    max_length = 1250
    train_features = convert_text_to_tensors(train_texts, processor, max_length)
    test_features = convert_text_to_tensors(test_texts, processor, max_length)

    # Create a neural network model
    # Modify the hyperparameters according to your model architecture
    vocab_size = len(processor.word_to_idx)
    embedding_dim = max_length
    hidden_size = 128
    output_size = 2  # Binary classification for sentiment analysis

    print("Variables:")
    print("\tTraining size\t", len(train_texts), "texts x", max_length, "tokens")
    print("\tTest size\t\t", len(test_texts), "texts x", max_length, "tokens")
    print("\tForward pass\t", embedding_dim, "embed ->", hidden_size, "hidden ->", output_size, "classes")

    model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)
    ne = 0
    outfile = 'trained_model.pth'
    if input("Load Model? 0|1\n") == "1":
        model.load_state_dict(torch.load(outfile))
        print("Model loaded.")
    else:
        print("Model reset.")
    try:
        ne = int(input("How many epochs?\n"))
    except:
        ne = 25
    finally:
        print("Okay, ", ne, "Epochs.")

    # Train
    training_history = train(model, train_features, train_labels, test_features, test_labels, ne)

    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)

    print()
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")

    print("\nMistaken Tests:")
    for i in evaluation_results['error_indexes']:
        print("\t- ", test_texts[i])

    # Save model weights to file
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")


