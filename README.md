# AI-fundamentals-Optimization-NLP
Foundational AI/ML exercises in Python — Stochastic Gradient Descent, Linear Classification with Hinge Loss, and an early-stage NLP sentiment data exploration. Built using NumPy, Math, and Pandas to understand core optimization concepts behind machine learning.

# AI Fundamentals: Optimization & NLP 

A set of foundational exercises from my Artificial Intelligence coursework, focused on understanding the **optimization techniques** that power machine learning models, plus an early step into NLP-based sentiment analysis.

## Libraries & Tools Used

- **NumPy** – vector/matrix operations, dot products, gradient calculations
- **Math** – square root calculations for adaptive learning rates
- **Pandas** – loading and exploring tabular/text data for the NLP task

No `scikit-learn`, `TensorFlow`, or `PyTorch` was used for the optimization algorithms — gradient descent and stochastic gradient descent were implemented manually from their mathematical definitions.

## What's inside

### 1. Stochastic Gradient Descent (SGD)
- Generated a synthetic regression dataset (1,000,000 examples) based on a known weight vector
- Implemented both **batch gradient descent** and **stochastic gradient descent** from scratch
- Used an adaptive learning rate (`η = 1/√numUpdates`) in SGD to stabilize convergence
- Compared training loss behavior between full-batch updates and per-example stochastic updates

### 2. Linear Classification with Hinge Loss
- Implemented a linear classifier trained using **hinge loss** (the loss function behind SVMs), built manually with NumPy
- Coded the gradient of hinge loss by hand and used gradient descent to optimize the weight vector
- Trained on a small 2D dataset to visualize how the decision boundary shifts as the model learns

### 3. NLP Sentiment Classification (Exploration Stage)
- Loaded a labeled text + sentiment dataset using Pandas
- Performed initial data exploration — inspecting structure, indexing rows, and iterating over text samples
- This notebook is a **work-in-progress** first step toward a full sentiment classification pipeline (vectorization + model training to be added next)

## Why I built these

These exercises were about understanding optimization at the level of the math, not the library call:
- How gradient descent minimizes a loss function step by step
- Why stochastic gradient descent trades exact gradients for speed at scale
- How hinge loss creates a margin-based decision boundary for classification
- How raw text data needs to be explored and cleaned before it can be turned into model-ready features

## Key takeaways

- Built two optimization algorithms (batch GD and SGD) entirely from their mathematical formulas
- Learned how learning rate scheduling affects convergence stability
- Got hands-on with margin-based loss functions used in classifiers like SVMs
- Started the foundation for an NLP sentiment classification pipeline

## Future Improvements

- Complete the NLP notebook with text vectorization (TF-IDF or Bag-of-Words) and a working classifier (Logistic Regression / Naive Bayes)
- Add an accuracy/F1-score evaluation for the sentiment model
- Visualize the decision boundary for the linear classifier
- Plot loss-vs-iteration curves for both GD and SGD for direct comparison

---

*Built as part of my Artificial Intelligence coursework, focused on understanding core optimization math before relying on ML libraries.*
