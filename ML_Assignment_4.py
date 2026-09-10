import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = data.data[:, [2]]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

degrees = [1, 2, 3, 5, 10]

train_mse = []
test_mse = []
train_r2 = []
test_r2 = []

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    train_mse.append(train_error)
    test_mse.append(test_error)

    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

print("\nPolynomial Regression Results")
print("-----------------------------------------------")
print("Degree\tTrain MSE\tTest MSE\tTrain R2\tTest R2")

for i in range(len(degrees)):
    print(
        degrees[i],
        "\t",
        round(train_mse[i], 2),
        "\t\t",
        round(test_mse[i], 2),
        "\t\t",
        round(train_r2[i], 3),
        "\t\t",
        round(test_r2[i], 3)
    )

best_index = np.argmin(test_mse)
best_degree = degrees[best_index]

print("\nBest Polynomial Degree:", best_degree)
print("Lowest Test MSE:", round(test_mse[best_index], 2))
print("Test R2 Score:", round(test_r2[best_index], 3))

plt.figure(figsize=(8, 5))

plt.plot(
    degrees,
    train_mse,
    marker='o',
    label='Training Error'
)

plt.plot(
    degrees,
    test_mse,
    marker='o',
    label='Testing Error'
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

X_plot = np.linspace(
    X.min(),
    X.max(),
    300
).reshape(-1, 1)

plt.scatter(
    X,
    y,
    alpha=0.3,
    label="Actual Data"
)

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_plot = model.predict(X_plot)

    plt.plot(
        X_plot,
        y_plot,
        label="Degree " + str(degree)
    )

plt.xlabel("BMI")
plt.ylabel("Diabetes Target")
plt.title("Polynomial Regression with Different Degrees")
plt.legend()
plt.grid(True)
plt.show()

print("\nBias-Variance Interpretation")
print("-----------------------------------------------")
print("Degree 1  : Simple model, high bias, low variance.")
print("Degree 2  : More flexible model, lower bias.")
print("Degree 3  : Moderate complexity and generally good balance.")
print("Degree 5  : More complex model, lower bias but higher variance.")
print("Degree 10 : Highly complex model, high variance and possible overfitting.")

print("\nAs polynomial degree increases:")
print("Model Complexity increases")
print("Bias decreases")
print("Variance increases")