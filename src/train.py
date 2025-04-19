from tensorflow.keras.callbacks import EarlyStopping
from src.models import build_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_model(X_train, y_train, model_type="lstm", task="all_labels"):
    if model_type == "lstm":
        X_train = X_train[..., np.newaxis]
        input_shape = X_train.shape[1:]
    else:
        input_shape = X_train.shape[1:]

    model = build_model(input_shape=input_shape, num_classes=len(set(y_train)), model_type=model_type)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, callbacks=[early_stop])
    return model, history

def evaluate_model(model, X_test, y_test, model_type="lstm"):
    if model_type == "lstm":
        X_test = X_test[..., np.newaxis]

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("🧪 Test Accuracy:", np.mean(y_pred_classes == y_test))
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
