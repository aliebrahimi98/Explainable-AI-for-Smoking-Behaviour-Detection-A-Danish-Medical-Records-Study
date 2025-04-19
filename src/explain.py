from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

def lime_explanation(model, X_test, task="all_labels", sentence_index=0):
    # Load corresponding raw text input
    df_test = pd.read_csv("data/pop2_merged.csv").dropna(subset=["text", "smoking_categorical", "PY_num_category"])
    df_test["label"] = df_test["smoking_categorical"] + "|" + df_test["PY_num_category"]
    df_test = df_test[~df_test["label"].isin(["Passive|High"])]
    df_test = df_test.reset_index(drop=True)

    # Fit tokenizer on test set text
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_test["text"])

    class_names = sorted(df_test["label"].unique().tolist())
    explainer = LimeTextExplainer(class_names=class_names)

    def make_predictions(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=100)
        padded = padded[..., np.newaxis]  # LSTM input
        return model.predict(padded)

    input_text = df_test["text"].iloc[sentence_index]
    print(f"📝 Explaining sentence {sentence_index}:{input_text}\n")

    explanation = explainer.explain_instance(
        input_text,
        classifier_fn=make_predictions,
        top_labels=1
    )

    # Show in notebook
    explanation.show_in_notebook(show_predicted_value=True)

    # Save as HTML
    explanation.save_to_file(f"lime_explanation_{sentence_index}.html")

    # Save as PNG
    fig = explanation.as_pyplot_figure()
    fig.savefig(f"lime_explanation_{sentence_index}.png", bbox_inches='tight')
