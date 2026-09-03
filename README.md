# Machine_Learning

## Linear Regression

This repository includes a small pure-Python linear regression implementation in
`linear_regression.py`.

### Example

```python
from linear_regression import LinearRegression

model = LinearRegression().fit([1, 2, 3, 4], [3, 5, 7, 9])
predictions = model.predict([5, 6])

print(model.intercept_)  # 1.0
print(model.coef_)       # [2.0]
print(predictions)       # [11.0, 13.0]
```