# Duplicate Question Detection using NLP and Deep Learning

## Overview

Online Q&A platforms receive thousands of new questions every day. Many users unknowingly ask questions that have already been answered. This leads to fragmented discussions and repeated answers.

The goal of this project is to automatically detect whether two questions have the **same meaning**, even if they are written differently. This is a classic Natural Language Processing (NLP) problem known as **semantic similarity** or **duplicate question detection**.

This project uses **Sentence-BERT embeddings and a neural network classifier** to determine whether two questions are duplicates.

---

# Objective

The objective of this project is to build a machine learning system that:

* Accepts two questions as input
* Understands their semantic meaning
* Predicts whether they represent the **same question**

Such a system can help platforms like Quora or StackOverflow automatically suggest existing questions instead of creating duplicates.

---

# Dataset

This project uses the **Quora Question Pairs dataset**.

Each record in the dataset contains:

* **question1** – First question
* **question2** – Second question
* **is_duplicate** – Label indicating whether the questions have the same meaning

Example:

| question1          | question2                        | is_duplicate |
| ------------------ | -------------------------------- | ------------ |
| What is AI?        | What is artificial intelligence? | 1            |
| How to cook pasta? | What is machine learning?        | 0            |

Dataset characteristics:

* Over **400,000 question pairs**
* Binary classification problem
* Real-world semantic similarity task

For faster experimentation, a **subset of the dataset (≈5000 samples)** was used during model development.

---

# Project Approach

The project follows a typical **machine learning pipeline**.

### 1. Data Preprocessing

* Removed missing values
* Selected relevant columns
* Sampled a smaller subset for faster computation

### 2. Train/Test Split

The dataset was divided into:

* **Training set (80%)**
* **Testing set (20%)**

This prevents data leakage and allows reliable evaluation.

### 3. Sentence Embeddings

Questions were converted into vector representations using:

**Sentence-BERT (all-MiniLM-L6-v2)**

Sentence embeddings capture the **semantic meaning of sentences**, allowing comparison between differently phrased questions.

Each question is converted into a **384-dimensional vector**.

### 4. Feature Engineering

Instead of using embeddings directly, additional features were created to help the model understand relationships between question pairs.

Features used:

* Embedding of question1
* Embedding of question2
* Absolute difference between embeddings
* Element-wise product of embeddings

This produces a final feature vector of **1536 dimensions**.

### 5. Feature Scaling

A **StandardScaler** was applied to normalize features, which improves neural network training stability.

### 6. Neural Network Classifier

A Multi-Layer Perceptron (MLP) was implemented using **PyTorch**.

Model architecture:

* Input layer: 1536 features
* Hidden layer: 256 neurons
* Hidden layer: 64 neurons
* Output layer: 1 neuron (duplicate probability)

Activation functions:

* ReLU
* Sigmoid (for final probability)

Dropout was used to reduce overfitting.

### 7. Model Training

The model was trained using:

* Binary Cross Entropy Loss
* Adam optimizer
* Multiple training epochs

### 8. Model Evaluation

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Results

The model successfully learns semantic similarity between question pairs.

Example prediction:

```
Input:
Q1: How can I learn Python quickly?
Q2: What is the fastest way to learn Python?

Output:
Duplicate Probability: 0.82
Prediction: Duplicate Question
```

The model performs well for paraphrased questions but may occasionally struggle with:

* negation ("not")
* subtle logical differences
* sarcasm or context-specific meaning

---

# Project Structure

```
duplicate-question-detection
│
├── dataset
│   └── quora_questions.csv
│
├── notebook
│   └── duplicate_detection.ipynb
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

Programming Language:

* Python

Libraries:

* pandas
* numpy
* scikit-learn
* sentence-transformers
* PyTorch
* matplotlib

---

# How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/your-username/duplicate-question-detection.git
cd duplicate-question-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the notebook

Open the Jupyter Notebook and execute the cells sequentially.

---

# Future Improvements

Several enhancements can improve this system:

### 1. Fine-tuning Transformer Models

Instead of using pre-trained embeddings, models like **BERT** or **RoBERTa** can be fine-tuned directly on the dataset for better performance.

### 2. Larger Training Dataset

Using more samples from the full dataset (~400k pairs) would improve generalization.

### 3. Better Feature Engineering

Additional features such as:

* word overlap
* sentence length difference
* syntactic similarity

could further improve performance.

### 4. Hyperparameter Optimization

Tuning learning rate, network size, and dropout could improve accuracy.

### 5. Real-Time Duplicate Search

The system could be extended to search a database of questions and return the most similar existing question.

### 6. Deployment

The model can be deployed as:

* a REST API
* a web application
* an integrated service in Q&A platforms

---

# Conclusion

This project demonstrates how **Natural Language Processing and Deep Learning** can be used to detect duplicate questions.

By combining **sentence embeddings with a neural network classifier**, the system can identify semantically similar questions even when their wording differs.

This approach can help improve the efficiency of online discussion platforms by reducing duplicate content and guiding users to existing answers.
