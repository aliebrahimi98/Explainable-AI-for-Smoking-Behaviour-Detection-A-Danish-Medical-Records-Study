import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(task="all_labels", max_len=100):
    # Load training and testing datasets
    train_df = pd.read_csv("data/pop1_merged.csv")
    test_df = pd.read_csv("data/pop2_merged.csv")

    # Create composite labels
    for df in [train_df, test_df]:
        df['label'] = df['smoking_categorical'] + "|" + df['PY_num_category']
        df.rename(columns={"text": "indhold"}, inplace=True)
        df.dropna(subset=["indhold", "label"], inplace=True)

    # Remove problematic class from training
    train_df = train_df[~train_df['label'].isin(['Passive|High'])]

    # Fit label encoder on train, apply to both
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_test = label_encoder.transform(test_df['label'])

    # Tokenize on training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['indhold'])

    X_train = tokenizer.texts_to_sequences(train_df['indhold'])
    X_test = tokenizer.texts_to_sequences(test_df['indhold'])

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test
