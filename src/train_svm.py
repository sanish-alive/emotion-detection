from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')

# Train SVM on extracted features
svm_model = SVC()
svm_model.fit(features, labels)

# Evaluate on validation data
val_features = model.predict(validation_generator)
val_predictions = svm_model.predict(val_features)
print("SVM Accuracy:", accuracy_score(validation_generator.labels, val_predictions))
