import unittest

from linear_regression import LinearRegression


class LinearRegressionTests(unittest.TestCase):
    def test_fit_and_predict_univariate_data(self) -> None:
        model = LinearRegression().fit([1, 2, 3, 4], [3, 5, 7, 9])

        self.assertAlmostEqual(model.intercept_, 1.0)
        self.assertEqual(len(model.coef_), 1)
        self.assertAlmostEqual(model.coef_[0], 2.0)
        self.assertEqual(model.predict([5, 6]), [11.0, 13.0])

    def test_fit_and_predict_multivariate_data(self) -> None:
        model = LinearRegression().fit(
            [[1, 1], [2, 0], [0, 2], [3, 1]],
            [6, 5, 7, 10],
        )

        self.assertAlmostEqual(model.intercept_, 1.0)
        self.assertEqual(len(model.coef_), 2)
        self.assertAlmostEqual(model.coef_[0], 2.0)
        self.assertAlmostEqual(model.coef_[1], 3.0)
        self.assertEqual(len(model.predict([[4, 2]])), 1)
        self.assertAlmostEqual(model.predict([[4, 2]])[0], 15.0)

    def test_predict_requires_a_fitted_model(self) -> None:
        with self.assertRaises(ValueError):
            LinearRegression().predict([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
