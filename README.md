# Multiclass Political Sentiment Analysis of Tamil X Comments Using Transformer-Based Hybrid Classification

## Abstract
Social media platforms have changed the entire sphere of political discourse with innumerable evaluations of public sentiments towards governance, policies, elections, and political leaders. This paper attempts Political Multiclass Sentiment Analysis on comments from Tamil X (Twitter) addressing issues including Tamil morphological richness, Agglutinative structure, code-mixing, slang, and sarcasm. In light of the limited annotated datasets available for Tamil political sentiment analysis, we developed a hybrid approach by combining IndicBERT (a transformer based model) and Random Forest Classifier. Our model is further enhanced by advanced preprocessing of structured Tamil political tweets. Our model achieved a macro F1 score of 0.2759 and ranked 14th  in the NAACL competitive benchmark. The pipeline was further improved after the deadline, leading to an improved F1 score of 0.3122.

## Key Features
- **Transformer-Based Model**: Utilizes IndicBERT for robust Tamil text representation.
- **Hybrid Classification Strategy**: Combines transformer embeddings with a Random Forest classifier for improved accuracy.
- **Advanced Preprocessing**: Handles text normalization, emoji retention, and code-mixed text filtering.
- **Multiclass Sentiment Categorization**: Classifies tweets into seven sentiment categories: Substantiated, Sarcastic, Opinionated, Positive, Negative, Neutral, and None of the Above.
- **Benchmark Performance**: Achieved a macro F1-score of **0.3122** (improved) and ranked **14th** (F1 score of **0.2759**) in the NAACL competitive benchmark.

## Dataset
The dataset consists of Tamil political tweets labeled across seven sentiment categories:
1. **Substantiated** - Remarks backed by factual data or references.
2. **Sarcastic** - Comments using irony or sarcasm.
3. **Opinionated** - Subjective views based on personal beliefs.
4. **Positive** - Supportive statements toward political leaders or policies.
5. **Negative** - Critical or unfavorable remarks.
6. **Neutral** - Objective or fact-based statements.
7. **None of the Above** - Tweets that do not fit any category.

## Methodology
### 1. Data Preprocessing
- Removed English words to maintain linguistic consistency.
- Applied script normalization and punctuation removal.
- Retained emojis as they contribute to sentiment expression.

### 2. Tokenization
- Used **IndicBERTv2-MLM-Sam-TLM** tokenizer with a sequence length of 512.
- Applied padding and truncation for consistency.

### 3. Model Selection
We experimented with multiple transformer models:
| Model | Language Coverage | Parameters | F1 Score |
|--------|------------------|------------|----------|
| DistilBERT | Multilingual (incl. Tamil) | 66M | 0.1580 |
| XLM-RoBERTa | Multilingual | 278M | 0.2027 |
| TamilBERT | Tamil only | 389M | 0.1037 |
| **IndicBERT** | Indic languages (incl. Tamil) | 470M | **0.2759** |

IndicBERT performed best and was selected as the base model.

### 4. Machine Learning Classifier Heads
We experimented with various classifiers on top of the transformer outputs:
- Random Forest (Best performer, F1-score: **0.3122**)
- Gaussian Naive Bayes
- Decision Tree
- SVM (Linear, RBF, Polynomial, Sigmoid)
- Gradient Boosting
- K-Nearest Neighbors
- Logistic Regression
- Multi-Layer Perceptron (MLP)

## Training Setup
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam (LR: 1e-6, Betas: (0.9, 0.999))
- **Learning Rate Scheduler**: ReduceLROnPlateau (Factor: 0.1, Patience: 4 epochs)
- **Evaluation Metrics**: Macro F1-score, Training & Testing Loss

## Results

![Base Model Comparison](Graphs/plot4.png)
*Comparison of different transformer-based base models for Tamil sentiment analysis.*

![Machine Learning Model Comparison](Graphs/plot1.png)
*Comparison of different machine learning models used as classification heads.*

- Achieved **macro F1-score: 0.2759** (Initial) and **0.3122** (Final improvement).
- Ranked **14th in NAACL competitive benchmark**.

## Contributors
- **Arun Prasad TD**
- **Ganesh Sundhar S**
- **Hari Krishnan N**
- **Shruthikaa V**
- **Sachin Kumar S**

## Acknowledgments
- NAACL organizers for the benchmark dataset.
- Hugging Face for pre-trained models.
- Amrita Vishwa Vidyapeetham for research support.



