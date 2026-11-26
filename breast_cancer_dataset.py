from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print("Dataset shape:", data.data.shape)

print("Feature names:")
print(data.feature_names)

print("Target names:")
print(data.target_names)

print("First 5 records:")
print(data.data[:5])

print("First 5 answers:")
print(data.target[:5])