import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import gradio as gr
import datetime

# -------------------- DOWNLOAD NLTK DATA --------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# -------------------- LOAD DATA --------------------
df = pd.read_csv("dataset.csv")  # Place dataset.csv in the same folder

# -------------------- TEXT CLEANING --------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_query'] = df['User Query'].apply(clean_text)
df['cleaned_output'] = df['Output'].apply(lambda x: x.lower().strip())

# -------------------- TRAIN / TEST SPLIT --------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -------------------- LOAD MODELS --------------------
print("Loading spaCy transformer model...")
nlp = spacy.load("en_core_web_trf")

print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------- ENCODE RESPONSES --------------------
responses = df['cleaned_output'].tolist()
response_embeddings = model.encode(responses)

# -------------------- RETRIEVAL FUNCTION --------------------
def retrieve_response(query):
    query_embedding = model.encode([query])
    cosine_scores = cosine_similarity(query_embedding, response_embeddings)
    best_match_idx = np.argmax(cosine_scores)
    return responses[best_match_idx]

# -------------------- PERSONALIZED EXERCISES --------------------
exercise_db = pd.DataFrame({
    "exercise_name": ["Push-ups", "Squats", "Deadlifts", "Plank", "Jump Rope"],
    "target_muscle": ["Chest, Triceps", "Legs, Glutes", "Back, Legs", "Core", "Cardio"],
    "difficulty_level": ["Beginner", "Beginner", "Advanced", "Beginner", "Intermediate"],
    "sets": [3, 3, 4, 3, "Timed"],
    "reps": [12, 15, 6, "Hold 30 sec", "60 sec"],
    "rest_time": ["30 sec", "30 sec", "60 sec", "N/A", "N/A"]
})

conversation_history = {}
query_logs = []

def get_exercise_suggestions(fitness_level):
    return exercise_db[exercise_db["difficulty_level"] == fitness_level].to_dict(orient="records")

def chatbot_response(user_id, user_query, mode, fitness_level, age, equipment):
    cleaned_query = clean_text(user_query)

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    if mode == "Personalized Query":
        enriched_query = (
            f"{cleaned_query} | fitness level: {fitness_level.lower()} | age: {age} | equipment: {', '.join(equipment)}"
        )
        final_query = enriched_query
    else:
        final_query = cleaned_query

    best_response = retrieve_response(final_query)
    conversation_history[user_id].append((user_query, best_response))

    query_logs.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "query": user_query,
        "response": best_response
    })

    exercises = get_exercise_suggestions(fitness_level)
    justification = (
        f"Since you are a {fitness_level.lower()} level trainee with {', '.join(equipment)}, "
        "these workouts are suitable for your level."
    )

    return best_response, conversation_history[user_id], exercises, justification

# -------------------- GRADIO INTERFACE --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🏋️ AI-Powered Fitness Chatbot")

    user_id = gr.Textbox(label="User ID", placeholder="Enter your unique ID")
    user_query = gr.Textbox(label="Your Query", placeholder="Ask your fitness question...")
    mode = gr.Radio(["Normal Query", "Personalized Query"], value="Normal Query")
    fitness_level = gr.Dropdown(["Beginner", "Intermediate", "Advanced"], label="Fitness Level")
    age = gr.Number(label="Age", value=25)
    equipment = gr.CheckboxGroup(["None", "Dumbbells", "Resistance Bands", "Kettlebell"], label="Available Equipment")

    output_response = gr.Textbox(label="Chatbot Response")
    output_history = gr.JSON(label="Conversation History")
    output_exercises = gr.JSON(label="Suggested Exercises")
    output_justification = gr.Textbox(label="Justification")

    submit_btn = gr.Button("Get Response")

    submit_btn.click(
        chatbot_response,
        inputs=[user_id, user_query, mode, fitness_level, age, equipment],
        outputs=[output_response, output_history, output_exercises, output_justification]
    )

if __name__ == "__main__":
    demo.launch()
