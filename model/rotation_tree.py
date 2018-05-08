import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class RotationTreeDecisionNode(object):
    def __init__(self, X, y, sample_weight, height=0):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.height = height

        self.child_nodes = None
        self.norm_vector = None
        self.rotation_point = None
        self.node_average = np.average(y, weights=sample_weight)

    @property
    def is_leaf(self):
        return self.child_nodes is None

    def can_be_split(self, min_leaf_size_to_split=None, max_height=None):
        # Only leaves can be split
        if not self.is_leaf:
            return False
        # We don't need to split leaves that contain only one unique value
        if np.unique(self.y).size < 2:
            return False
        # Leaf is too small to split
        if min_leaf_size_to_split and self.y.size < min_leaf_size_to_split:
            return False
        # Tree is too big to split
        if max_height and self.height >= max_height:
            return False
        return True

    def clean_train_data(self):
        self.X = None
        self.y = None
        self.sample_weight = None

    def split(self):
        new_y = self.y - self.node_average
        neg_rows = (new_y < 0).ravel()
        pos_rows = ~neg_rows
        neg_center = np.average(self.X[neg_rows],
                                weights=np.multiply(self.sample_weight[neg_rows].ravel(),
                                                    np.abs(new_y[neg_rows].ravel())),
                                axis=0)
        pos_center = np.average(self.X[pos_rows],
                                weights=np.multiply(self.sample_weight[pos_rows].ravel(),
                                                    np.abs(new_y[pos_rows].ravel())),
                                axis=0)
        if np.array_equal(neg_center, pos_center):
            # Tie breaker for rare cases when we have absolutely aligned classes
            self.norm_vector = np.ones(shape=pos_center.shape)
            self.rotation_point = pos_center
        else:
            self.norm_vector = pos_center - neg_center
            self.rotation_point = (pos_center * np.dot(self.sample_weight[neg_rows].ravel(),
                                                       np.abs(new_y[neg_rows].ravel())) +
                                   neg_center * np.dot(self.sample_weight[pos_rows].ravel(),
                                                       np.abs(new_y[pos_rows].ravel()))) / np.dot(
                self.sample_weight.ravel(), np.abs(new_y.ravel()))
        split_results = np.dot(self.X - self.rotation_point, self.norm_vector)
        left_child_rows = (split_results < 0).ravel()
        right_child_nodes = ~left_child_rows
        left_child = RotationTreeDecisionNode(X=self.X[left_child_rows],
                                              y=self.y[left_child_rows],
                                              sample_weight=self.sample_weight[left_child_rows],
                                              height=self.height + 1)
        right_child = RotationTreeDecisionNode(X=self.X[right_child_nodes],
                                               y=self.y[right_child_nodes],
                                               sample_weight=self.sample_weight[
                                                   right_child_nodes],
                                               height=self.height + 1)

        return left_child, right_child

    def apply(self, X):
        if self.is_leaf:
            return np.zeros(shape=(X.shape[0], 1)) + self.node_average
        split_results = np.dot(X - self.rotation_point, self.norm_vector)
        left_child_rows = (split_results < 0).ravel()
        right_child_nodes = ~left_child_rows
        result = np.zeros(shape=(X.shape[0], 1))
        result[left_child_rows] = self.child_nodes[0].apply(X[left_child_rows])
        result[right_child_nodes] = self.child_nodes[1].apply(X[right_child_nodes])
        return result


class RotationTreeBase(BaseEstimator):
    def __init__(self, max_height=None, min_leaf_size_to_split=None):
        self.max_height = max_height
        self.min_leaf_size_to_split = min_leaf_size_to_split
        self.decision_nodes = []
        self.split_pointer = -1

    @property
    def is_fitted(self):
        return self.split_pointer == len(self.decision_nodes)

    @property
    def height(self):
        return max(node.height for node in self.decision_nodes)

    def fit(self, X, y, sample_weight=None):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if sample_weight is None:
            sample_weight = np.ones(shape=y.shape)
        root = RotationTreeDecisionNode(X, y, sample_weight)
        self.decision_nodes.append(root)
        self.split_pointer = 0
        while self.split_pointer < len(self.decision_nodes):
            if self.decision_nodes[self.split_pointer].can_be_split(
                    min_leaf_size_to_split=self.min_leaf_size_to_split,
                    max_height=self.max_height
            ):
                lt_child, gte_child = self.decision_nodes[self.split_pointer].split()
                self.decision_nodes.append(lt_child)
                self.decision_nodes.append(gte_child)
                self.decision_nodes[
                    self.split_pointer].child_nodes = (lt_child, gte_child)
            self.decision_nodes[self.split_pointer].clean_train_data()
            self.split_pointer += 1

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError('Attempt to predict with unfitted RotationTree')
        return self.decision_nodes[0].apply(X)


class RotationTreeClassifier(RotationTreeBase, ClassifierMixin):
    def predict_proba(self, X):
        return super(RotationTreeClassifier, self).predict(X)

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores >= 0.5).astype(np.int)


class RotationTreeRegressor(RotationTreeBase, RegressorMixin):
    pass
