import numpy as np

import random

import matplotlib.pyplot as plt
import matplotlib

import copy

import pandas
from pandas import DataFrame

from typing import NoReturn, Tuple, List

# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    path_to_csv : str
        Путь к cancer датасету.
    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).
    """
    lables_and_features: DataFrame = pandas.read_csv(filepath_or_buffer=path_to_csv).sample(frac=1).reset_index(drop=True)

    lables: np.array = np.array([1 if label == 'M' else 0 for label in lables_and_features['label']])
    features: np.array = np.array(lables_and_features.drop('label', axis=1))

    return features, lables

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.
    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.
    """
    lables_and_features: DataFrame = pandas.read_csv(filepath_or_buffer=path_to_csv).sample(frac=1).reset_index(drop=True)

    lables: np.array = np.array(lables_and_features['label'])
    features: np.array = np.array(lables_and_features.drop('label', axis=1))

    return features, lables

# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.
    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.
    """
    train_size: int = int(X.shape[0] * ratio)
    test_size: int = X.shape[0] - train_size

    indices = np.random.permutation(X.shape[0])

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test

# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """
    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.
    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).
    """
    num_classes: List = len(np.unique(list(y_pred) + list(y_true)))
    tp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for pred, true in zip(y_pred, y_true):
        tn[np.logical_and(np.arange(len(tn)) != pred, np.arange(len(tn)) != true)] += 1
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = sum(tp) / len(y_pred)

    return precision, recall, accuracy

# Task 4

class KDTree:

    class Item:
        def __init__(self, axis: int, data: np.array, key: float):
            self.axis: int = axis
            self.key: float = key
            self.data: np.array = data

        def find(self, point: np.array) -> bool:
            return point[self.axis] > self.key

    class Node:
        def __init__(self, axis: int, data: np.array, key: float):
            self.item: KDTree.Item = KDTree.Item(axis, data, key)
            self.right: KDTree.Node = None
            self.left: KDTree.Node = None

        def find(self, point: np.array) -> bool:
            return self.item.find(point)

    class Pair:
        def __init__(self, data: np.array, indice: int):
            self.data: np.array = data
            self.indice: int = indice

    def __init__(self, X: np.array, leaf_size: int = 40):
        """
        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).
        Returns
        -------
        """
        self.leaf_size: int = leaf_size
        self.points = X
        self.dimension: int = X.shape[1]
        self.kd_three: KDTree.Node = self.create_kd_three(np.array([KDTree.Pair(x, i) for i, x in enumerate(X)]))

    def create_kd_three(self, X: np.array, axis:int = 0) -> Node:
        median = np.median([pair.data[axis] for pair in X])

        left: np.array = X[[pair.data[axis] < median for pair in X]]
        right: np.array = X[[pair.data[axis] >= median for pair in X]]

        if left.size < self.leaf_size or right.size < self.leaf_size:
            return KDTree.Node(axis, X, median)

        node: KDTree.Node = KDTree.Node(axis, None, median)

        node.left = self.create_kd_three(left, (axis + 1) % self.dimension)
        node.right = self.create_kd_three(right, (axis + 1) % self.dimension)

        return node

    def take_k_nearbors_points(self, X: np.array, x: np.array, k: int = 1) -> List[Pair]:
        points: np.array = np.array([point.data for point in X])

        distances = np.linalg.norm(points - x, axis=1)
        indices = np.argsort(distances)

        k_nearest_indices = indices[:k]
        k_nearest_points = [X[i] for i in k_nearest_indices]

        return k_nearest_points

    def take_max_distance_for_k_nearbors_points(self, X: np.array, x: np.array) -> float:
        points: np.array = np.array([point.data for point in X])

        distances = np.linalg.norm(points - x, axis=1)
        indices = np.argsort(distances)

        max_distance: float = indices[-1:]

        return distances[max_distance]

    def find(self, x: np.array, kd_three: Node, k: int = 1, axis: int = 0) -> List:
        if kd_three.item.data is not None:
            return self.take_k_nearbors_points(kd_three.item.data, x, k)

        if kd_three.find(x):
            nearest_points: List = self.find(x, kd_three.right, k, (axis + 1) % self.dimension)

            point_to_age_disnatce: float = abs(kd_three.item.key - x[kd_three.item.axis])
            max_distance_for_k_nearbors_points: float = self.take_max_distance_for_k_nearbors_points(nearest_points, x)

            if point_to_age_disnatce > max_distance_for_k_nearbors_points:
                return nearest_points

            return self.take_k_nearbors_points(nearest_points + self.find(x, kd_three.left, k, (axis + 1) % self.dimension), x, k)
        else:
            nearest_points: List = self.find(x, kd_three.left, k, (axis + 1) % self.dimension)

            point_to_age_disnatce: float = abs(kd_three.item.key - x[kd_three.item.axis])
            max_distance_for_k_nearbors_points: float = self.take_max_distance_for_k_nearbors_points(nearest_points, x)

            if point_to_age_disnatce > max_distance_for_k_nearbors_points:
                return nearest_points

            return self.take_k_nearbors_points(nearest_points + self.find(x, kd_three.right, k, (axis + 1) % self.dimension), x, k)

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.
        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.
        """
        result: List[List] = []

        for x in X:
            points_indexs: List = self.find(x, self.kd_three, k)
            result.append([point_index.indice for point_index in points_indexs])

        return result

# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """
        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.
        """
        self.n_neighbors: int = n_neighbors
        self.leaf_size: int = leaf_size
        self.kd_tree: KDTree = None
        self.labels: np.array = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.
        """
        self.kd_tree = KDTree(X, self.leaf_size)
        self.labels = y

    def predict_proba(self, X: np.array) -> List[np.array]:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
        """
        predicts = []
        num_classes = len(np.unique(list(self.labels)))
        nearbors_indices: List[List] = self.kd_tree.query(X, self.n_neighbors)

        for nearbor_indices in nearbors_indices:
            class_probabilities: List = [0] * num_classes
            for i in range(num_classes):
                for index in nearbor_indices:
                    if self.labels[index] == i:
                        class_probabilities[i] += 1
            predicts.append(class_probabilities)

        return predicts

    def predict(self, X: np.array) -> np.array:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        Returns
        -------
        np.array
            Вектор предсказанных классов.
        """
        return np.argmax(self.predict_proba(X), axis=1)
