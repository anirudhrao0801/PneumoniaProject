import os
from data_preparation import prepare_data
from model_definition import create_cnn_model
from training import train_model, evaluate_model, plot_accuracy, plot_loss

def main():
    # Check available data
    print(os.listdir("../input"))
    print(os.listdir('../input/chest_xray/chest_xray'))

    # Prepare data
    training_set, validation_generator, test_set = prepare_data()

    # Create model
    cnn = create_cnn_model()
    cnn.summary()

    # Train model
    cnn_model = train_model(cnn, training_set, validation_generator)

    # Evaluate model
    evaluate_model(cnn, test_set)

    # Plot results
    plot_accuracy(cnn_model)
    plot_loss(cnn_model)

if __name__ == "__main__":
    main()
