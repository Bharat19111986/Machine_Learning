"""Small pure-Python linear regression implementation."""

from __future__ import annotations

from collections.abc import Iterable


class LinearRegression:
    def __init__(self) -> None:
        self.intercept_ = 0.0
        self.coef_: list[float] = []
        self._is_fitted = False

    def fit(
        self,
        features: Iterable[float] | Iterable[Iterable[float]],
        targets: Iterable[float],
    ) -> "LinearRegression":
        rows = _normalize_features(features)
        y_values = [float(value) for value in targets]

        if not rows:
            raise ValueError("features must not be empty")
        if len(rows) != len(y_values):
            raise ValueError("features and targets must have the same length")

        design_matrix = [[1.0, *row] for row in rows]
        xt = _transpose(design_matrix)
        coefficients = _solve_linear_system(
            _multiply_matrices(xt, design_matrix),
            _multiply_matrix_vector(xt, y_values),
        )

        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]
        self._is_fitted = True
        return self

    def predict(
        self,
        features: Iterable[float] | Iterable[Iterable[float]],
    ) -> list[float]:
        if not self._is_fitted:
            raise ValueError("model must be fitted before prediction")

        rows = _normalize_features(features)
        expected_feature_count = len(self.coef_)

        for row in rows:
            if len(row) != expected_feature_count:
                raise ValueError("feature count does not match fitted model")

        return [
            self.intercept_
            + sum(coefficient * value for coefficient, value in zip(self.coef_, row))
            for row in rows
        ]


def _normalize_features(
    features: Iterable[float] | Iterable[Iterable[float]],
) -> list[list[float]]:
    rows = list(features)
    if not rows:
        return []

    first_row = rows[0]
    if isinstance(first_row, Iterable) and not isinstance(first_row, (str, bytes)):
        normalized_rows = [[float(value) for value in row] for row in rows]
    else:
        normalized_rows = [[float(value)] for value in rows]

    feature_count = len(normalized_rows[0])
    if feature_count == 0:
        raise ValueError("each feature row must contain at least one value")
    if any(len(row) != feature_count for row in normalized_rows):
        raise ValueError("all feature rows must have the same length")

    return normalized_rows


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _multiply_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_transposed = _transpose(right)
    return [
        [sum(left_value * right_value for left_value, right_value in zip(row, column)) for column in right_transposed]
        for row in left
    ]


def _multiply_matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(value * weight for value, weight in zip(row, vector)) for row in matrix]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("cannot fit a singular feature matrix")

        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_index:
                continue

            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[row_index][-1] for row_index in range(size)]
