#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def sigmoid(x: float) -> float:
    return 1/(1+np.exp(-x))

def dsigmoid(x: float) -> float:
    sig = sigmoid(x)
    return sig*(1-sig)

def accuracy(data: np.ndarray,
             weights: np.ndarray,
             target: np.ndarray,
             boundary_decision=0) -> float:
    target_pred = (data @ weights > boundary_decision).astype(int)
    bad_count = np.sum(np.abs(target_pred - target))

    return 1 - bad_count / target.shape[0]

def loss(data: np.ndarray,
             weights: np.ndarray,
             target: np.ndarray) -> float:
    sum_ = 0
    for i in range(data.shape[0]):
        yi = data[i].T @ weights
        if target[i] == 1:
            sum_ += -np.log(sigmoid(yi))
        else:
            sum_ += -np.log(1-sigmoid(yi))


    return sum_ / data.shape[0]


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    ones = np.ones((data.shape[0], 1))
    data = np.hstack((data, ones))

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed
    )

    # Generate initial logistic regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        for batch_offset in range(0, train_data.shape[0], args.batch_size):
            sum_ = 0
            for rel_offset in range(args.batch_size):
                i = permutation[batch_offset + rel_offset]
                yi = train_data[i].T @ weights
                if train_target[i] == 1:
                    s = -1/sigmoid(yi) * dsigmoid(yi) * train_data[i].T
                else:
                    s = 1/(1-sigmoid(yi)) * dsigmoid(yi) * train_data[i].T

                sum_ += s

            g = sum_ / args.batch_size
            weights -= args.learning_rate*g


        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.

        train_accuracy = accuracy(train_data, weights, train_target)
        train_loss = loss(train_data, weights, train_target)
        test_accuracy = accuracy(test_data, weights, test_target)
        test_loss = loss(test_data, weights, test_target)

        print(train_accuracy)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=20, cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, label="train", marker="P", cmap="RdBu")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
