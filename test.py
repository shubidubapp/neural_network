# This data is not full of it and
# It is taken from here : https://pjreddie.com/projects/mnist-in-csv/

from nnetwork import NeuralNetwork
import pickle
from numpy import interp

train_data = "data/mnist_train_10000.csv"
test_data = "data/mnist_test_1000.csv"

input_count = 28*28
hidden_node = 30
output_node = 10

learning_rate = 0.1

save_file = "mnist_nnetwork.pkl"


def data_reader(data):
    with open(data) as t:
        for line in t:
            values = line.strip().split(",")
            label_value = int(values[0])
            targets = [0 for _ in range(output_node)]
            targets[label_value] = 0.99
            inputs = values[1:]
            inputs = (interp(inputs, [0, 256], [0, 0.99]) + 0.01).tolist()
            yield targets, inputs


def main(train_over_save_file=False, guess=False, save_to_file=False):
    if guess:
        with open(save_file, "rb") as f:
            network = pickle.load(f)
        for target, inputs in data_reader(train_data):
            target_value = target.index(max(target))
            guess = network.guess(inputs)
            guess_value = guess.index(max(guess))
            error = (target_value - guess_value) / 10 * 100
            print("Target: {}, Guess: {}, Error: {}"
                  .format(target_value, guess_value, error))
            answer = input("Wanna guess another ? [Y,n]: ")
            if answer and answer[0].lower() == "n":
                return

    else:
        if train_over_save_file:
            with open(save_file, "rb") as f:
                network = pickle.load(f)
        else:
            network = NeuralNetwork(input_count, [hidden_node], output_node, learning_rate)
        train_count = 0
        for target, inputs in data_reader(train_data):
            network.train(inputs, target)
            if train_count % 100 == 0:
                target_value = target.index(max(target))
                guess = network.guess(inputs)
                guess_value = guess.index(max(guess))
                error = (target_value - guess_value)/10 * 100
                print("Train count: {}, Target: {}, Guess: {}, Error: {}"
                      .format(train_count, target_value, guess_value, error))
            train_count += 1
        if save_to_file:
            with open(save_file, "wb") as f:
                pickle.dump(network, f)

main()
