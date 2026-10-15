#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

TEST_SIZE = 0.2
SEED = 42

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        dataset = Dataset()


        print(dataset)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            dataset.data, dataset.target, test_size=TEST_SIZE, random_state=SEED
        )

        cat_cols = []
        num_cols = []

        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            if (bool(np.all(col == col.astype(int)))):
                cat_cols.append(i)
            else:
                num_cols.append(i)

        enc = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        scaler = sklearn.preprocessing.StandardScaler()
        ct = sklearn.compose.ColumnTransformer(
            [
                ("categorical", enc, cat_cols),
                ("numerical", scaler, num_cols)

            ]
        )

        pipeline = sklearn.pipeline.Pipeline(
            [
                ("col_transform", ct),
                # ("poly", poly)
            ]
        )

        train_data = pipeline.fit_transform(X_train)
        test_data = pipeline.transform(X_test)

        y_pred = test_data @ y_train
        rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)
        print(f"RMSE: {rmse}")



        # TODO: Train a model on the given dataset and store it in `model`.
        model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = ...

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
