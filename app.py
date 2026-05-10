import gradio as gr
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load model and embedder ───────────────────────────────────────────────────
print("Loading model...")
embedder = SentenceTransformer("model/embedder")

with open("model/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

print("✅ Ready!")

# ── Prediction function ───────────────────────────────────────────────────────
def predict_sentiment(review):
    if not review.strip():
        return "Please enter a movie review."
    
    # Embed the input text
    embedding = embedder.encode([review])
    
    # Predict
    prediction = classifier.predict(embedding)[0]
    
    # Try to get confidence score
    try:
        proba = classifier.decision_function(embedding)[0]
        confidence = round(float(abs(proba)) * 10, 1)
        confidence = min(confidence, 99.9)  # cap at 99.9%
    except:
        confidence = None
    
    if prediction == 1:
        label = "🟢 POSITIVE"
        message = "This review expresses a positive opinion about the movie."
    else:
        label = "🔴 NEGATIVE"
        message = "This review expresses a negative opinion about the movie."
    
    if confidence:
        return f"{label}\n\n{message}\n\nConfidence score: {confidence}"
    return f"{label}\n\n{message}"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste a movie review here... e.g. 'This film was an absolute masterpiece!'",
        label="Movie Review"
    ),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="🎬 Movie Review Sentiment Classifier",
    description=(
        "This classifier uses **sentence embeddings** (all-MiniLM-L6-v2) + **Linear SVM** "
        "to predict whether a movie review is positive or negative.\n\n"
        "Trained on a filtered IMDB dataset. Accuracy: **84.5%** on test set."
    ),
    examples=[
        ["This movie was absolutely brilliant! The acting was superb and the story kept me hooked throughout."],
        ["Complete waste of time. The plot made no sense and the characters were utterly boring."],
        ["A decent film, nothing special but enjoyable enough for a lazy Sunday afternoon."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()