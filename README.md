# ExerciseXpertv1
# 🏋 AI-Powered Fitness Chatbot with Semantic Search

This project is a fitness-focused conversational AI that uses NLP techniques, transformer models, and semantic search to retrieve the most relevant workout recommendations for user queries. It also supports personalized exercise suggestions based on fitness level, equipment availability, and user preferences.

## 🚀 Features

*Data Preprocessing:*
- Text cleaning (lowercasing, punctuation removal, stopword removal, lemmatization) using NLTK.
- Dataset splitting into training and test sets.

*Transformer-based NLP:*
- Uses spaCy (en_core_web_trf) for tokenization, POS tagging, dependency parsing, and named entity recognition (NER).
- Generates semantic embeddings with Sentence-BERT (all-MiniLM-L6-v2).

*Semantic Search:*
- Retrieves the best matching response for a query using cosine similarity.
- Evaluated using Mean Reciprocal Rank (MRR) and Recall@k.

*Personalization:*
- Suggests workouts tailored to user’s fitness level, equipment, and age.
- Multi-turn conversation memory.

*Interactive Interface:*
- Integrated Gradio web app for easy user interaction.
- Collects user feedback for improvement.

## 📂 Project Structure

├── dataset.csv               # Fitness Q&A dataset
├── nlp_lab_project.ipynb     # Main Jupyter/Colab notebook
├── requirements.txt          # Required Python dependencies
├── app.py                    # (Optional) Script to run the Gradio interface
└── README.md                 # Project documentation


## 🛠 Installation

Clone the repository:
bash
git clone https://github.com/yourusername/fitness-chatbot.git
cd fitness-chatbot

Install dependencies:
bash
pip install -r requirements.txt

Download required NLP models:
bash
python -m spacy download en_core_web_trf


## ▶ Usage

*Run in Colab*
- Upload the dataset and notebook to Google Colab.
- Install dependencies inside Colab and run all cells.

*Run Locally*
bash
python app.py

Then open the provided Gradio link in your browser.

## 📊 Evaluation
- Mean Reciprocal Rank (MRR): 0.507
- Recall@3: 0.548

## 📦 Technologies Used
- Python 3.11
- Pandas, NumPy
- NLTK
- spaCy (Transformer-based model)
- Sentence-BERT (all-MiniLM-L6-v2)
- scikit-learn
- Gradio

## 🖼 Example Interaction
*User:* "I need a high-intensity workout for my legs."

*Chatbot:* "Incorporate exercises like high knees, butt kicks, and leg swings to activate muscles and prepare your body for high-intensity training."

## 📌 Future Improvements
- Integrate Hugging Face API for faster embeddings retrieval.
- Expand dataset with more fitness topics (nutrition, recovery, mental health).
- Add voice input/output for hands-free interaction.
- Deploy on Hugging Face Spaces or Streamlit Cloud.

## 🤝 Contributing
Contributions are welcome! Please fork the repo and create a pull request.

## 📜 License
This project is licensed under the MIT License.
