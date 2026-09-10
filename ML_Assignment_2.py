import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\PRAJWAL\Downloads\Breast_Cancer.csv")

df = df.dropna()

X = df.drop("Status", axis=1)
y = df["Status"]

X = pd.get_dummies(X, drop_first=True)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

start_time = time.time()

model.fit(X_train_std, y_train)

y_pred_base = model.predict(X_test_std)

end_time = time.time()

base_accuracy = accuracy_score(y_test, y_pred_base)
base_execution_time = end_time - start_time

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

start_time = time.time()

model.fit(X_train_pca, y_train)

y_pred_pca = model.predict(X_test_pca)

end_time = time.time()

pca_accuracy = accuracy_score(y_test, y_pred_pca)
pca_execution_time = end_time - start_time

c = len(set(y))

lda = LinearDiscriminantAnalysis(n_components=c - 1)

X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

start_time = time.time()

model.fit(X_train_lda, y_train)

y_pred_lda = model.predict(X_test_lda)

end_time = time.time()

lda_accuracy = accuracy_score(y_test, y_pred_lda)
lda_execution_time = end_time - start_time

results = pd.DataFrame({
    "Method": ["Baseline", "PCA", "LDA"],
    "Features": [
        X_train_std.shape[1],
        X_train_pca.shape[1],
        X_train_lda.shape[1]
    ],
    "Accuracy": [
        base_accuracy,
        pca_accuracy,
        lda_accuracy
    ],
    "Execution Time": [
        base_execution_time,
        pca_execution_time,
        lda_execution_time
    ]
})

print("\nComparison Results:")
print(results)

results.plot(
    x="Method",
    y="Accuracy",
    kind="bar",
    title="Accuracy Comparison"
)

plt.xlabel("Method")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

results.plot(
    x="Method",
    y="Execution Time",
    kind="bar",
    title="Execution Time Comparison"
)

plt.xlabel("Method")
plt.ylabel("Time in Seconds")
plt.show()