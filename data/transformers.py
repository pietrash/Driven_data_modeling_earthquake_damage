from statistics import mode

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin


class FillUnkWithMode(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """
    Fills missing values in a DataFrame with the mode of each column.

    This transformer fills missing values in a DataFrame with the mode (most frequent value) 
    of each column. It fits on the training data to calculate the mode of each column 
    and then transforms data using the calculated modes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unique_values = None
        self.modes = None

    def fit(self, X, y=None):
        """
        Fit the transformer on input data.

        Parameters:
        X : pandas.DataFrame
            Input data to fit the transformer.

        Returns:
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)

        self.unique_values = {}
        self.modes = {}
        for column in X.columns:
            self.unique_values[column] = set(X[column])
            self.modes[column] = mode(X[column])

        return self

    def transform(self, X):
        """
        Transform input data.

        Parameters:
        X : pandas.DataFrame
            Input data to transform.

        Returns:
        out : pandas.DataFrame
            Transformed data with missing values filled using mode.
        """
        assert isinstance(X, pd.DataFrame)
        assert self.unique_values is not None
        assert self.modes is not None

        out = X.copy()

        for column in out.columns:
            out[column] = out.apply(
                lambda x: x[column] if x[column] in self.unique_values[column] else self.modes[column],
                axis=1
            )

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        #     print(out)

        return out


class TargetProbability(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """
    Calculates the probability of a target value for each unique value in specified columns.

    This transformer calculates the probability of a target value for each unique value in the specified 
    columns. It fits on the training data to compute the probabilities and then transforms data to 
    add new features containing these probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prob_dict = None
        self.target_values = None

    def fit(self, X, y):
        """
        Fit the transformer on input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.
        y : pandas.DataFrame
            Target data.

        Returns:
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        self.prob_dict = {}
        self.target_values = set(y)
        for column in X.columns:
            self.prob_dict[column] = {}

            grouped = pd.concat([X, y], axis=1).groupby(column)[y.name]
            for value, group in grouped:
                total_count = len(group)
                probabilities = {}
                for target, count in group.value_counts().items():
                    probabilities[target] = count / total_count
                self.prob_dict[column][value] = probabilities

        return self

    def transform(self, X):
        """
        Transform input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.

        Returns:
        out : pandas.DataFrame
            Transformed data with added probability features.
        """
        assert isinstance(X, pd.DataFrame)
        assert self.prob_dict != {}

        out = pd.DataFrame()
        for column in X.columns:
            for value in self.target_values:
                out[f'{column}_{value}_probability'] = X.apply(
                    lambda x: self.prob_dict.get(column, {}).get(x[column], {}).get(value, 0),
                    axis=1
                )

        return out


class FrequencyEncoding(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """
    Encodes categorical features by their frequency counts.

    This transformer encodes categorical features by replacing them with their frequency counts 
    in the training data. It fits on the training data to compute the frequency counts for each 
    unique value in each column and then transforms data to 
    add new features containing these frequency counts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.freq_dict = None

    def fit(self, X, y=None):
        """
        Fit the transformer on input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.

        Returns:
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)

        self.freq_dict = {}
        for column in X.columns:
            self.freq_dict[column] = dict(X[column].value_counts())
        return self

    def transform(self, X):
        """
        Transform input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.

        Returns:
        out : pandas.DataFrame
            Transformed data with added frequency encoding features.
        """
        assert isinstance(X, pd.DataFrame)
        assert self.freq_dict is not None
        assert set(X.columns) == set(self.freq_dict.keys())

        out = pd.DataFrame()
        for column in X.columns:
            out[column + '_frequency'] = X.apply(
                lambda x: self.freq_dict[column].get(x[column], None),
                axis=1
            )
        return out


class TargetMeanGrouped(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """
    Computes the mean target value for groups of columns.

    This transformer calculates the mean target value for groups of columns in the input data. 
    It fits on the training data to compute the mean target values for each unique combination 
    of column values, and then transforms both training and test data to add new features 
    containing these mean target values. If a new combination of values is encountered in 
    the test data, the mode of the mean target values from the training data is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.default_mean_value = None
        self.mean_dict = None

    def fit(self, X, y):
        """
        Fit the transformer on input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.
        y : pandas.Series
            Target data.

        Returns:
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        self.mean_dict = dict(pd.concat([X, y], axis=1).groupby(X.columns.to_list())[y.name].mean())
        self.default_mean_value = mode(self.mean_dict.values())

        return self

    def transform(self, X):
        """
        Transform input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.

        Returns:
        out : pandas.DataFrame
            Transformed data with added mean target value features.
        """
        assert isinstance(X, pd.DataFrame)
        assert self.mean_dict is not None

        out = pd.DataFrame()
        out['_'.join(X.columns.to_list()) + '_mean'] = X.apply(
            lambda x: self.mean_dict.get(tuple(x[col] for col in X.columns.to_list()), self.default_mean_value),
            axis=1
        )

        return out


class TargetMean(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    """
    Computes the mean target value for each unique value in the specified columns.

    This transformer calculates the mean target value for each unique value in the specified 
    columns in the input data. It fits on the training data to compute the mean target values 
    for each unique value in each column, and then transforms data to
    add new features containing these mean target values.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean_dict = None

    def fit(self, X, y):
        """
        Fit the transformer on input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.
        y : pandas.Series
            Target data.

        Returns:
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        self.mean_dict = {}
        for column in X.columns:
            self.mean_dict[column] = dict(pd.concat([X, y], axis=1).groupby(column)[y.name].mean())

        return self

    def transform(self, X):
        """
        Transform input data.

        Parameters:
        X : pandas.DataFrame
            Input features data.

        Returns:
        out : pandas.DataFrame
            Transformed data with added mean target value features.
        """
        assert isinstance(X, pd.DataFrame)

        out = pd.DataFrame()
        for column in X.columns:
            out[column + '_mean'] = X.apply(
                lambda x: self.mean_dict.get(column, {}).get(x[column]),
                axis=1
            )

        return out
