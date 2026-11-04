"""
LearnMate - Agentic AI for Personalized Course Pathways
Problem Statement #12
Backend: Python Flask + IBM Granite (ibm-watsonx-ai SDK)
"""

import os
import json
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise RuntimeError("SECRET_KEY is not configured.")

# ── Production security settings ───────────────────────────────────────────────
# Cookies are HttpOnly (no JS access), SameSite=Lax (CSRF mitigation), and
# Secure when FLASK_DEBUG=0 so they are only sent over HTTPS in production.
app.config["DEBUG"] = False  # explicit guard — never rely on env var alone
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Enable Secure flag only in production (when not running in debug/HTTP dev mode)
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_DEBUG", "0") == "0"

# ── IBM Granite client (lazy-initialized) ──────────────────────────────────────
_granite_client = None

def get_granite_client():
    global _granite_client
    api_key    = session.get("ibm_api_key")    or os.getenv("IBM_API_KEY", "")
    project_id = session.get("ibm_project_id") or os.getenv("IBM_PROJECT_ID", "")
    url        = session.get("ibm_url")        or os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        return None, None

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        credentials = Credentials(url=url, api_key=api_key)
        client = APIClient(credentials, project_id=project_id)
        # Pin API version — prevents SDK from auto-generating a future date
        client.version = "2024-05-31"
        return client, project_id
    except Exception as e:
        print(f"[IBM] Client init error: {e}")
        return None, None


def ask_granite(user_message: str, profile: dict) -> str:
    """Call IBM Granite via ibm-watsonx-ai SDK. Falls back to rich knowledge base on error."""
    client, project_id = get_granite_client()
    model_id = session.get("ibm_model", os.getenv("IBM_MODEL", "ibm/granite-3-3-8b-instruct"))

    if client:
        try:
            from ibm_watsonx_ai.foundation_models import ModelInference

            system_prompt = (
                f"You are LearnMate, an AI learning coach powered by IBM Granite.\n"
                f"Student: {profile.get('name','Student')}, "
                f"Track: {profile.get('track','Data Science')}, "
                f"Level: {profile.get('level','Intermediate')} ({profile.get('progress',0)}% complete), "
                f"Goal: {profile.get('goal','Data Scientist')}.\n"
                f"Skills: {', '.join(f'{k}: {v}%' for k,v in profile.get('skills',{}).items())}.\n"
                f"Provide personalized, encouraging, actionable guidance. Use markdown for structure."
            )

            model = ModelInference(
                model_id=model_id,
                api_client=client,
                project_id=project_id,
            )
            # Use the current chat API (generate_text is deprecated)
            response = model.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                params={
                    "max_tokens": 700,
                    "temperature": 0,
                },
            )
            result = response["choices"][0]["message"]["content"]
            return result.strip()
        except Exception as e:
            print(f"[IBM Granite] generation error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  COMPREHENSIVE KNOWLEDGE BASE — covers small→large concepts across topics
    # ═══════════════════════════════════════════════════════════════════════════
    msg   = user_message.lower()
    name  = profile.get("name", "there")
    track = profile.get("track", "Data Science")
    level = profile.get("level", "Intermediate")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TYPES OF ML / SUPERVISED / UNSUPERVISED / REINFORCEMENT
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["types of ml", "types of machine", "supervised", "unsupervised",
                               "reinforcement", "semi-supervised", "what is ml",
                               "what is machine learning", "branches of ml"]):
        return (
            f"Here are the **Types of Machine Learning** — from basics to advanced:\n\n"
            "**1. Supervised Learning** (most common)\n"
            "- Learns from *labelled* data (input → correct output)\n"
            "- Sub-types: **Regression** (predict numbers) & **Classification** (predict categories)\n"
            "- Algorithms: Linear Regression, Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost\n"
            "- Examples: House price prediction, email spam detection, disease diagnosis\n\n"
            "**2. Unsupervised Learning**\n"
            "- Finds patterns in *unlabelled* data without correct answers\n"
            "- Sub-types: **Clustering**, **Dimensionality Reduction**, **Association**\n"
            "- Algorithms: K-Means, DBSCAN, PCA, Apriori\n"
            "- Examples: Customer segmentation, anomaly detection, topic modeling\n\n"
            "**3. Reinforcement Learning**\n"
            "- Agent learns by *trial and error* (reward / penalty feedback)\n"
            "- Algorithms: Q-Learning, DQN, PPO, A3C\n"
            "- Examples: Game-playing AI (AlphaGo), robotics, self-driving cars\n\n"
            "**4. Semi-Supervised Learning**\n"
            "- Small labelled + large unlabelled data\n"
            "- Useful when labelling is expensive (medical imaging, speech)\n\n"
            "**5. Self-Supervised Learning** (modern)\n"
            "- Model creates its own labels from raw data\n"
            "- Used in GPT, BERT, IBM Granite LLMs\n\n"
            f"At **{level}** level: master Supervised Learning first — it covers 80% of real-world DS jobs!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ML ALGORITHMS — comprehensive list
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["algorithm", "algorithms", "random forest", "decision tree",
                               "svm", "support vector", "knn", "k-nearest", "naive bayes",
                               "logistic regression", "linear regression", "gradient boost",
                               "xgboost", "lightgbm", "adaboost", "bagging", "boosting",
                               "ensemble", "scikit"]):
        return (
            f"**Complete ML Algorithms Guide** for your **{track}** track:\n\n"
            "--- REGRESSION (predict continuous values) ---\n"
            "- **Linear Regression** — straight-line fit; y = mx + c\n"
            "- **Polynomial Regression** — curved relationship\n"
            "- **Ridge / Lasso** — regularization to prevent overfitting\n"
            "- **SVR** — Support Vector Regression\n\n"
            "--- CLASSIFICATION (predict categories) ---\n"
            "- **Logistic Regression** — sigmoid curve, binary/multiclass\n"
            "- **Decision Tree** — if-else rules; highly interpretable\n"
            "- **Random Forest** — 100s of trees averaged; very accurate\n"
            "- **SVM** — max-margin hyperplane between classes\n"
            "- **KNN** — vote from k nearest neighbors\n"
            "- **Naive Bayes** — probability-based; great for text\n"
            "- **Gradient Boosting / XGBoost / LightGBM** — top Kaggle performers\n\n"
            "--- CLUSTERING (unsupervised grouping) ---\n"
            "- **K-Means** — partition into k clusters\n"
            "- **DBSCAN** — density-based; handles noise well\n"
            "- **Hierarchical** — builds a tree of clusters\n\n"
            "--- DIMENSIONALITY REDUCTION ---\n"
            "- **PCA** — Principal Component Analysis; reduce features\n"
            "- **t-SNE** — visualize high-dimensional data in 2D\n"
            "- **Autoencoders** — neural network-based reduction\n\n"
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import classification_report\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
            "model.fit(X_train, y_train)\n"
            "print(classification_report(y_test, model.predict(X_test)))\n"
            "```\n\n"
            "**Start with:** Linear Regression → Logistic Regression → Decision Tree → Random Forest"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. CLASSIFICATION vs REGRESSION vs CLUSTERING
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["classification", "regression", "clustering", "difference between"]):
        return (
            "**Classification vs Regression vs Clustering** — key differences:\n\n"
            "| Concept | Output | Example |\n"
            "|---|---|---|\n"
            "| Classification | Category label | Spam/Not Spam, Cat/Dog |\n"
            "| Regression | Continuous number | House price, Temperature |\n"
            "| Clustering | Groups (no labels) | Customer segments |\n\n"
            "**Classification** — *Is this email spam?* → Yes / No\n"
            "- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC\n"
            "- Algorithms: Logistic Regression, SVM, Random Forest, KNN\n\n"
            "**Regression** — *What will the house sell for?* → Rs. 45,00,000\n"
            "- Metrics: MAE, MSE, RMSE, R² Score\n"
            "- Algorithms: Linear Regression, Ridge, Lasso, SVR, XGBoost\n\n"
            "**Clustering** — *Which customers are similar?* → Group A, B, C\n"
            "- No labels needed; discover natural groupings\n"
            "- Algorithms: K-Means, DBSCAN, Hierarchical Clustering\n\n"
            "**Quick Rule:** If output is a *category* → Classification. "
            "If output is a *number* → Regression. "
            "If there are *no labels* → Clustering."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. DEEP LEARNING & NEURAL NETWORKS
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["deep learning", "neural network", "ann", "cnn", "rnn",
                               "lstm", "gru", "tensorflow", "pytorch", "keras",
                               "convolutional", "recurrent", "activation function",
                               "backpropagation", "epoch", "batch size", "dropout",
                               "overfitting neural"]):
        return (
            "**Deep Learning — Small to Large Concepts:**\n\n"
            "--- BASICS ---\n"
            "- **Neuron** — basic unit; takes inputs, applies weights + bias, passes through activation\n"
            "- **Activation Functions:** ReLU (most used), Sigmoid (0-1), Tanh (-1 to 1), Softmax (multiclass)\n"
            "- **Layer types:** Input → Hidden → Output\n"
            "- **Weights & Biases** — learned parameters; updated via gradient descent\n"
            "- **Backpropagation** — algorithm to compute gradients and update weights\n"
            "- **Epoch** — one full pass through training data\n"
            "- **Batch Size** — number of samples per weight update\n"
            "- **Learning Rate** — how fast the model updates weights\n\n"
            "--- NETWORK ARCHITECTURES ---\n"
            "- **ANN** — Fully connected; general purpose\n"
            "- **CNN** — Convolutional layers; best for images (image classification, object detection)\n"
            "- **RNN** — Recurrent; processes sequences (time series, text)\n"
            "- **LSTM / GRU** — Advanced RNN; solves vanishing gradient; used in language modeling\n"
            "- **Transformer** — Attention mechanism; basis for GPT, BERT, IBM Granite\n"
            "- **Autoencoder** — Encoder-decoder; used for compression, anomaly detection\n"
            "- **GAN** — Generator vs Discriminator; creates realistic images/text\n\n"
            "--- REGULARIZATION ---\n"
            "- **Dropout** — randomly zero out neurons during training to prevent overfitting\n"
            "- **Batch Normalization** — normalize layer inputs; faster training\n"
            "- **L1/L2 Regularization** — penalize large weights\n\n"
            "```python\n"
            "import tensorflow as tf\n\n"
            "model = tf.keras.Sequential([\n"
            "    tf.keras.layers.Dense(256, activation='relu', input_shape=(20,)),\n"
            "    tf.keras.layers.Dropout(0.3),\n"
            "    tf.keras.layers.Dense(128, activation='relu'),\n"
            "    tf.keras.layers.Dense(10, activation='softmax')  # 10-class output\n"
            "])\n"
            "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
            "model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)\n"
            "```\n\n"
            "**Resources:** fast.ai (free), deeplearning.ai (Coursera), TensorFlow tutorials"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. NLP — Natural Language Processing
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["nlp", "natural language", "text processing", "tokeniz",
                               "stemming", "lemmatiz", "pos tagging", "ner",
                               "named entity", "sentiment", "word2vec", "tfidf", "tf-idf",
                               "bag of words", "text classification", "text mining"]):
        return (
            "**NLP — Natural Language Processing (Small to Large):**\n\n"
            "--- BASICS ---\n"
            "- **Tokenization** — split text into words/sentences\n"
            "- **Stop Word Removal** — remove 'the', 'is', 'at' etc.\n"
            "- **Stemming** — reduce 'running' → 'run' (crude)\n"
            "- **Lemmatization** — reduce 'running' → 'run' (linguistically correct)\n"
            "- **POS Tagging** — label noun, verb, adjective etc.\n"
            "- **NER** — Named Entity Recognition (find PERSON, ORG, LOCATION)\n\n"
            "--- TEXT REPRESENTATION ---\n"
            "- **Bag of Words (BoW)** — word frequency count vector\n"
            "- **TF-IDF** — Term Frequency × Inverse Document Frequency; weights important words\n"
            "- **Word2Vec** — word embeddings; 'king' - 'man' + 'woman' = 'queen'\n"
            "- **GloVe / FastText** — alternative embedding methods\n"
            "- **BERT Embeddings** — contextual; same word has different meaning in different contexts\n\n"
            "--- NLP TASKS ---\n"
            "- **Text Classification** — spam, sentiment, topic categorization\n"
            "- **Machine Translation** — English to Marathi, Hindi etc.\n"
            "- **Question Answering** — extract answer from a passage\n"
            "- **Text Summarization** — abstractive vs extractive\n"
            "- **Chatbots** — intent detection + response generation\n\n"
            "```python\n"
            "import nltk\n"
            "from nltk.tokenize import word_tokenize\n"
            "from nltk.corpus import stopwords\n\n"
            "text = 'LearnMate helps students learn Data Science effectively.'\n"
            "tokens = word_tokenize(text.lower())\n"
            "clean = [w for w in tokens if w not in stopwords.words('english')]\n"
            "print(clean)  # ['learnmate', 'helps', 'students', 'learn', 'data', 'science']\n"
            "```\n\n"
            "**Libraries:** NLTK, spaCy, HuggingFace Transformers, Gensim"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 6. LLM — Large Language Models
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["llm", "large language model", "gpt", "bert", "chatgpt",
                               "transformer model", "attention mechanism", "prompt",
                               "fine-tun", "rag", "retrieval augmented", "generative ai",
                               "foundation model", "ibm granite", "llama", "mistral"]):
        return (
            "**LLMs — Large Language Models (Small to Large):**\n\n"
            "--- WHAT IS AN LLM? ---\n"
            "A Large Language Model is a deep learning model trained on *massive text data* "
            "to understand and generate human language.\n\n"
            "--- KEY CONCEPTS ---\n"
            "- **Transformer Architecture** — basis of all modern LLMs\n"
            "  - Self-Attention: model focuses on relevant words in context\n"
            "  - Multi-Head Attention: multiple attention heads in parallel\n"
            "  - Positional Encoding: encodes word order\n"
            "- **Pre-training** — train on huge corpus (books, web, code)\n"
            "- **Fine-tuning** — adapt pre-trained model to specific task\n"
            "- **RLHF** — Reinforcement Learning from Human Feedback (makes ChatGPT helpful)\n"
            "- **Prompt Engineering** — craft inputs to get better outputs\n"
            "- **RAG (Retrieval-Augmented Generation)** — LLM + external knowledge base\n"
            "- **Tokens** — LLMs process text as tokens (1 token ≈ 0.75 words)\n"
            "- **Context Window** — max tokens the model can 'see' at once\n\n"
            "--- POPULAR LLMs ---\n"
            "| Model | By | Specialty |\n"
            "|---|---|---|\n"
            "| IBM Granite 3.3 | IBM | Enterprise, safe, open-source |\n"
            "| GPT-4 / ChatGPT | OpenAI | General purpose, coding |\n"
            "| BERT | Google | Understanding tasks (NLP) |\n"
            "| LLaMA 3 | Meta | Open-source, customizable |\n"
            "| Gemini | Google | Multimodal (text + images) |\n"
            "| Mistral | Mistral AI | Lightweight, fast |\n\n"
            "--- USING IBM GRANITE (LearnMate uses this!) ---\n"
            "```python\n"
            "from ibm_watsonx_ai.foundation_models import ModelInference\n\n"
            "model = ModelInference(model_id='ibm/granite-3-3-8b-instruct', ...)\n"
            "response = model.generate_text(prompt='Explain overfitting in ML:')\n"
            "print(response)\n"
            "```\n\n"
            "--- CAREER PATHS ---\n"
            "- **LLM Engineer** — fine-tune and deploy LLMs\n"
            "- **Prompt Engineer** — design optimal prompts\n"
            "- **AI/ML Engineer** — build RAG pipelines, AI apps\n\n"
            "**Key insight:** IBM Granite powering LearnMate IS an LLM — you're already using this tech!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 7. STATISTICS — complete guide
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["statistic", "probability", "distribution", "hypothesis",
                               "p-value", "mean", "median", "mode", "variance",
                               "standard deviation", "normal distribution", "central limit",
                               "bayesian", "correlation", "covariance", "z-score",
                               "t-test", "anova", "chi-square", "confidence interval",
                               "sample", "population", "inferential", "descriptive"]):
        return (
            "**Statistics for Data Science — Small to Large:**\n\n"
            "--- DESCRIPTIVE STATISTICS (summarize data) ---\n"
            "- **Mean** — average: sum / count\n"
            "- **Median** — middle value; robust to outliers\n"
            "- **Mode** — most frequent value\n"
            "- **Variance (σ²)** — average squared deviation from mean\n"
            "- **Standard Deviation (σ)** — sqrt of variance; same unit as data\n"
            "- **Range** — max - min\n"
            "- **IQR** — Interquartile Range (Q3 - Q1); used in box plots\n"
            "- **Skewness** — asymmetry of distribution\n"
            "- **Kurtosis** — tail heaviness\n\n"
            "--- PROBABILITY ---\n"
            "- P(A) = favourable / total; P(A) ∈ [0,1]\n"
            "- **Conditional Probability:** P(A|B) = P(A∩B) / P(B)\n"
            "- **Bayes Theorem:** P(A|B) = P(B|A) × P(A) / P(B)\n"
            "- **Independence:** P(A∩B) = P(A) × P(B)\n\n"
            "--- PROBABILITY DISTRIBUTIONS ---\n"
            "- **Normal (Gaussian)** — bell curve; mean=0, std=1 (standard normal)\n"
            "- **Binomial** — n trials, p success probability\n"
            "- **Poisson** — count events in fixed time (calls per hour)\n"
            "- **Uniform** — all values equally likely\n"
            "- **Exponential** — time between events\n\n"
            "--- INFERENTIAL STATISTICS (draw conclusions from samples) ---\n"
            "- **Hypothesis Testing:** H₀ (null) vs H₁ (alternative)\n"
            "- **p-value:** probability of observing data if H₀ is true\n"
            "  - p < 0.05 → reject H₀ (statistically significant)\n"
            "- **t-test** — compare means (1-sample, 2-sample, paired)\n"
            "- **ANOVA** — compare means of 3+ groups\n"
            "- **Chi-Square** — test independence of categorical variables\n"
            "- **Confidence Interval** — range likely containing true parameter (95% CI)\n\n"
            "--- CORRELATION & REGRESSION ---\n"
            "- **Pearson r** — linear correlation (-1 to +1)\n"
            "- **Spearman ρ** — rank-based correlation (non-linear)\n"
            "- **Covariance** — direction of relationship (not scaled)\n\n"
            "```python\n"
            "import numpy as np\n"
            "from scipy import stats\n\n"
            "data = [23, 25, 28, 22, 30, 27, 24]\n"
            "print('Mean:', np.mean(data))       # 25.57\n"
            "print('Std:', np.std(data))         # 2.69\n"
            "print('Median:', np.median(data))   # 25.0\n\n"
            "# t-test: is mean different from 25?\n"
            "t, p = stats.ttest_1samp(data, 25)\n"
            "print('p-value:', round(p, 4))      # 0.56 (not significant)\n"
            "```\n"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 8. SQL — comprehensive guide
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["sql", "mysql", "postgresql", "sqlite", "database",
                               "query", "select", "join", "group by", "having",
                               "where", "order by", "subquery", "cte", "window function",
                               "stored procedure", "index", "primary key", "foreign key",
                               "normalization", "nosql", "mongodb"]):
        return (
            "**SQL — Small to Large Concepts:**\n\n"
            "--- BASICS ---\n"
            "- **SELECT** — retrieve columns\n"
            "- **WHERE** — filter rows\n"
            "- **ORDER BY** — sort results\n"
            "- **LIMIT** — restrict number of rows\n"
            "- **DISTINCT** — remove duplicates\n\n"
            "--- AGGREGATIONS ---\n"
            "- COUNT(), SUM(), AVG(), MIN(), MAX()\n"
            "- **GROUP BY** — group rows for aggregation\n"
            "- **HAVING** — filter groups (like WHERE for aggregated data)\n\n"
            "--- JOINS ---\n"
            "- **INNER JOIN** — rows matching in both tables\n"
            "- **LEFT JOIN** — all from left + matching from right\n"
            "- **RIGHT JOIN** — all from right + matching from left\n"
            "- **FULL OUTER JOIN** — all rows from both tables\n"
            "- **SELF JOIN** — join a table with itself\n\n"
            "--- ADVANCED ---\n"
            "- **Subqueries** — query inside a query\n"
            "- **CTEs (WITH clause)** — named temporary result set\n"
            "- **Window Functions:** ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD(), SUM() OVER()\n"
            "- **Indexes** — speed up queries (B-tree, Hash)\n"
            "- **Normalization** — 1NF, 2NF, 3NF (eliminate redundancy)\n"
            "- **Transactions** — COMMIT, ROLLBACK, ACID properties\n"
            "- **Stored Procedures** — saved SQL logic\n\n"
            "```sql\n"
            "-- Window function example\n"
            "SELECT\n"
            "    name,\n"
            "    department,\n"
            "    salary,\n"
            "    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,\n"
            "    AVG(salary) OVER (PARTITION BY department) AS dept_avg\n"
            "FROM employees;\n\n"
            "-- CTE example\n"
            "WITH top_earners AS (\n"
            "    SELECT *, RANK() OVER (ORDER BY salary DESC) AS rnk FROM employees\n"
            ")\n"
            "SELECT * FROM top_earners WHERE rnk <= 5;\n"
            "```\n\n"
            "**NoSQL (MongoDB):** stores JSON-like documents; no fixed schema\n"
            "```python\n"
            "from pymongo import MongoClient\n"
            "db = MongoClient()['mydb']\n"
            "db.students.insert_one({'name': 'Alex', 'score': 95})\n"
            "results = db.students.find({'score': {'$gte': 90}})\n"
            "```\n\n"
            "**Practice:** SQLZoo, LeetCode SQL, Mode Analytics, HackerRank SQL"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 9. POWER BI
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["power bi", "powerbi", "power query", "dax", "power pivot",
                               "business intelligence", "bi tool", "report", "dashboard tool"]):
        return (
            "**Power BI — Small to Large Concepts:**\n\n"
            "--- WHAT IS POWER BI? ---\n"
            "Microsoft's Business Intelligence tool to connect data, transform it, "
            "and create interactive visual dashboards — no coding needed!\n\n"
            "--- COMPONENTS ---\n"
            "- **Power BI Desktop** — build reports (free download)\n"
            "- **Power BI Service** — publish & share online\n"
            "- **Power BI Mobile** — view dashboards on phone\n\n"
            "--- DATA WORKFLOW ---\n"
            "1. **Connect** — Excel, SQL Server, CSV, Azure, Google Sheets, APIs\n"
            "2. **Transform** — Power Query Editor (clean & reshape data)\n"
            "3. **Model** — define relationships between tables (Star Schema)\n"
            "4. **Visualize** — drag & drop charts, maps, KPIs\n"
            "5. **Publish** — share with team / embed in web\n\n"
            "--- KEY CONCEPTS ---\n"
            "- **Power Query (M language)** — ETL: Extract, Transform, Load\n"
            "- **Data Model** — tables + relationships (like a mini database)\n"
            "- **DAX (Data Analysis Expressions)** — formula language for calculated columns & measures\n"
            "- **Measures** — dynamic calculations (SUM, AVERAGE, CALCULATE, FILTER)\n"
            "- **Slicers** — interactive filters for end users\n"
            "- **Drill-through** — click to see detailed data\n\n"
            "--- IMPORTANT DAX FORMULAS ---\n"
            "```\n"
            "Total Sales = SUM(Sales[Amount])\n"
            "Sales YTD = TOTALYTD(SUM(Sales[Amount]), Dates[Date])\n"
            "Profit % = DIVIDE([Profit], [Total Sales], 0)\n"
            "Running Total = CALCULATE(SUM(Sales[Amount]),\n"
            "    FILTER(ALL(Dates), Dates[Date] <= MAX(Dates[Date])))\n"
            "```\n\n"
            "--- VISUAL TYPES ---\n"
            "Bar/Column, Line, Pie/Donut, Scatter, Map, Treemap, Waterfall, Funnel, KPI Card, Matrix\n\n"
            "--- BEST PRACTICES ---\n"
            "- Use Star Schema (fact + dimension tables)\n"
            "- Avoid many-to-many relationships\n"
            "- Use measures over calculated columns for performance\n"
            "- Keep visuals simple — max 5-7 per page\n\n"
            "**Certification:** PL-300 (Microsoft Power BI Data Analyst)"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 10. EXCEL — small to large
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["excel", "spreadsheet", "vlookup", "pivot table", "macro",
                               "vba", "index match", "xlookup", "sumif", "countif",
                               "conditional formatting", "excel formula", "excel chart"]):
        return (
            "**Excel for Data Analysis — Small to Large:**\n\n"
            "--- BASICS ---\n"
            "- **Cells, Rows, Columns** — A1 notation; Ctrl+C/V/Z\n"
            "- **Basic formulas:** =SUM(), =AVERAGE(), =COUNT(), =MAX(), =MIN()\n"
            "- **AutoFill** — drag to copy formula\n"
            "- **Absolute Reference:** $A$1 (doesn't change when copied)\n"
            "- **Named Ranges** — give meaningful names to cell ranges\n\n"
            "--- INTERMEDIATE FORMULAS ---\n"
            "- **IF:** =IF(A1>50, 'Pass', 'Fail')\n"
            "- **VLOOKUP:** =VLOOKUP(A2, Table, 3, 0) — search & return value\n"
            "- **INDEX-MATCH:** more flexible than VLOOKUP\n"
            "- **XLOOKUP:** modern replacement for VLOOKUP (Office 365)\n"
            "- **SUMIF/COUNTIF:** sum/count with a condition\n"
            "- **SUMIFS/COUNTIFS:** multiple conditions\n"
            "- **TEXT functions:** LEFT(), RIGHT(), MID(), LEN(), TRIM(), CONCAT()\n"
            "- **DATE functions:** TODAY(), NOW(), DATEDIF(), YEAR(), MONTH()\n\n"
            "--- ADVANCED ---\n"
            "- **Pivot Tables** — summarize large datasets with drag & drop\n"
            "  - Row/Column/Value/Filter fields\n"
            "  - Calculated Fields & Items\n"
            "  - Slicers & Timelines\n"
            "- **Pivot Charts** — charts linked to Pivot Tables\n"
            "- **Power Query** — import, clean, transform data (no code)\n"
            "- **Conditional Formatting** — color cells by rules/data bars/icon sets\n"
            "- **Data Validation** — dropdown lists, restrict input\n"
            "- **What-If Analysis:** Goal Seek, Scenario Manager, Data Tables\n"
            "- **Solver** — optimization (like linear programming)\n"
            "- **Macros & VBA** — automate repetitive tasks with code\n\n"
            "```vba\n"
            "Sub HighlightBlanks()\n"
            "    Dim cell As Range\n"
            "    For Each cell In Selection\n"
            "        If IsEmpty(cell) Then cell.Interior.Color = RGB(255, 0, 0)\n"
            "    Next cell\n"
            "End Sub\n"
            "```\n\n"
            "**Excel for DS:** Use for quick EDA, pivot tables for summary stats, "
            "Power Query for data cleaning before moving to Python/Power BI.\n\n"
            "**Certification:** MO-200 (Excel Associate), MO-201 (Excel Expert)"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 11. R PROGRAMMING
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["r programming", "r language", "rstudio", "ggplot", "dplyr",
                               "tidyverse", "r for data", "caret", "tidyr", "r vs python"]):
        return (
            "**R Programming for Data Science — Small to Large:**\n\n"
            "--- WHAT IS R? ---\n"
            "R is a statistical programming language created specifically for data analysis, "
            "statistics, and visualization. Widely used in academia, healthcare, and research.\n\n"
            "--- R vs PYTHON ---\n"
            "| Aspect | R | Python |\n"
            "|---|---|---|\n"
            "| Strength | Statistics & visualization | General purpose + ML |\n"
            "| Syntax | More mathematical | More readable |\n"
            "| ML Libraries | caret, tidymodels | scikit-learn, TensorFlow |\n"
            "| Visualization | ggplot2 (excellent) | matplotlib, seaborn |\n"
            "| Job Market | Academia, biostat, pharma | Industry, tech, DS |\n\n"
            "--- BASICS ---\n"
            "```r\n"
            "# Vectors & basic operations\n"
            "x <- c(10, 20, 30, 40, 50)\n"
            "mean(x)    # 30\n"
            "sd(x)      # 15.81\n"
            "summary(x) # min, Q1, median, mean, Q3, max\n\n"
            "# Data Frame\n"
            "df <- data.frame(name=c('Alice','Bob'), score=c(90,75))\n"
            "df[df$score > 80, ]  # filter rows\n"
            "```\n\n"
            "--- TIDYVERSE (modern R) ---\n"
            "```r\n"
            "library(dplyr)\n"
            "library(ggplot2)\n\n"
            "# dplyr pipeline\n"
            "df %>%\n"
            "    filter(score > 70) %>%\n"
            "    group_by(department) %>%\n"
            "    summarise(avg_score = mean(score)) %>%\n"
            "    arrange(desc(avg_score))\n\n"
            "# ggplot2 visualization\n"
            "ggplot(df, aes(x=age, y=salary, color=dept)) +\n"
            "    geom_point() + geom_smooth(method='lm') +\n"
            "    theme_minimal() + labs(title='Age vs Salary')\n"
            "```\n\n"
            "--- KEY R PACKAGES ---\n"
            "- **dplyr** — data manipulation (filter, select, mutate, summarise)\n"
            "- **ggplot2** — beautiful visualizations (grammar of graphics)\n"
            "- **tidyr** — reshape data (pivot_longer, pivot_wider)\n"
            "- **caret / tidymodels** — machine learning\n"
            "- **lubridate** — date/time handling\n"
            "- **stringr** — string manipulation\n"
            "- **readr** — fast CSV reading\n"
            "- **shiny** — build interactive web apps in R\n\n"
            "**Best for:** Statistical analysis, academic research, clinical trials, financial modeling"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 12. ENGLISH COMMUNICATION FOR DATA SCIENCE
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["english", "communication", "presentation", "storytelling",
                               "report writing", "explain results", "business writing",
                               "email writing", "soft skill"]):
        return (
            f"**English Communication Skills for Data Scientists:**\n\n"
            "As a Data Scientist, {name}, technical skills alone aren't enough — "
            "you must communicate insights clearly to non-technical stakeholders!\n\n"
            "--- DATA STORYTELLING ---\n"
            "- **What happened?** (Descriptive — show the data)\n"
            "- **Why did it happen?** (Diagnostic — explain the cause)\n"
            "- **What will happen?** (Predictive — model results)\n"
            "- **What should we do?** (Prescriptive — recommendation)\n\n"
            "--- PRESENTING RESULTS ---\n"
            "- Start with the **business problem**, not the technical method\n"
            "- Use visuals — replace tables with charts\n"
            "- Say: *'Our model predicts 87% accuracy'* not *'F1=0.87, AUC-ROC=0.93'*\n"
            "- End with a **clear recommendation** and next steps\n\n"
            "--- REPORT WRITING STRUCTURE ---\n"
            "1. Executive Summary (1 paragraph — key finding)\n"
            "2. Problem Statement\n"
            "3. Data & Methodology\n"
            "4. Key Findings (with charts)\n"
            "5. Recommendations\n"
            "6. Appendix (technical details)\n\n"
            "--- USEFUL PHRASES ---\n"
            "- *'The data suggests that...'*\n"
            "- *'Our analysis indicates a strong correlation between...'*\n"
            "- *'Based on the model results, we recommend...'*\n"
            "- *'The key takeaway from this analysis is...'*\n"
            "- *'Further investigation is needed to...'*\n\n"
            "--- EMAIL TO STAKEHOLDER TEMPLATE ---\n"
            "```\n"
            "Subject: Customer Churn Analysis — Key Findings\n\n"
            "Hi [Name],\n\n"
            "I've completed the churn analysis for Q3.\n"
            "Key finding: Customers inactive for 30+ days are 3x more likely to churn.\n\n"
            "Recommendation: Launch a re-engagement campaign for this segment.\n"
            "Expected impact: 15% reduction in monthly churn rate.\n\n"
            "Happy to walk through the details in a meeting.\n"
            "Best, Alex\n"
            "```\n\n"
            "**Tip:** Practice explaining your ML models in plain English daily — it's a superpower in interviews!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 13. MARATHI — language support
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["marathi", "mराठी", "marathi language", "learn marathi",
                               "marathi translation", "data science in marathi",
                               "explain in marathi", "marathi medium"]):
        return (
            f"नमस्कार {name}! 🙏 LearnMate मध्ये आपले स्वागत आहे!\n\n"
            "**Data Science — मराठीमध्ये सोप्या शब्दांत:**\n\n"
            "**Machine Learning म्हणजे काय?**\n"
            "Machine Learning म्हणजे संगणकाला उदाहरणांवरून शिकवणे. "
            "जसे आपण अनुभवातून शिकतो, तसे संगणक डेटावरून शिकतो.\n\n"
            "**Supervised Learning (पर्यवेक्षित शिक्षण):**\n"
            "- उत्तरे सांगून शिकवणे\n"
            "- उदाहरण: 'हा ईमेल spam आहे' असे सांगितले तर model शिकतो\n"
            "- उपयोग: रोग निदान, किंमत अंदाज\n\n"
            "**Unsupervised Learning (अपर्यवेक्षित शिक्षण):**\n"
            "- उत्तरे न देता स्वतः patterns शोधणे\n"
            "- उदाहरण: ग्राहकांचे गट करणे (clustering)\n\n"
            "**Python Library मराठीत:**\n"
            "- `pandas` — डेटा हाताळणे (Excel सारखे)\n"
            "- `numpy` — गणितीय गणना\n"
            "- `matplotlib` — आलेख (graphs) काढणे\n"
            "- `scikit-learn` — ML models बनवणे\n\n"
            "**महत्त्वाच्या संकल्पना:**\n"
            "- Dataset — माहितीचा संच\n"
            "- Training — मॉडेलला शिकवणे\n"
            "- Testing — मॉडेलची चाचणी\n"
            "- Accuracy — अचूकता\n"
            "- Overfitting — जास्त शिकणे (नवीन डेटावर चुकणे)\n\n"
            "**Data Science चे करिअर संधी:**\n"
            "Data Scientist, ML Engineer, Data Analyst, Business Intelligence Analyst\n\n"
            "English मध्ये विचारलेल्या प्रश्नांवरही मी मदत करतो. 😊\n"
            "*(For full Marathi AI responses, connect your IBM Granite API key!)*"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 14. PYTHON for Data Science
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["python", "pandas", "numpy", "matplotlib", "seaborn",
                               "scikit-learn", "sklearn", "dataframe", "data frame",
                               "list comprehension", "lambda", "dictionary", "function"]):
        return (
            "**Python for Data Science — Small to Large:**\n\n"
            "--- BASICS ---\n"
            "```python\n"
            "# Variables & Data Types\n"
            "name = 'Alex'          # str\n"
            "score = 95             # int\n"
            "gpa = 3.8              # float\n"
            "is_enrolled = True     # bool\n"
            "subjects = ['Math', 'ML', 'SQL']  # list\n"
            "info = {'name': 'Alex', 'age': 21} # dict\n\n"
            "# List Comprehension\n"
            "squares = [x**2 for x in range(10)]\n\n"
            "# Lambda Function\n"
            "double = lambda x: x * 2\n"
            "```\n\n"
            "--- PANDAS (data wrangling) ---\n"
            "```python\n"
            "import pandas as pd\n\n"
            "df = pd.read_csv('data.csv')\n"
            "df.head()              # first 5 rows\n"
            "df.info()              # column types & nulls\n"
            "df.describe()          # summary statistics\n"
            "df['col'].value_counts()   # frequency count\n"
            "df.dropna()            # remove nulls\n"
            "df.fillna(0)           # fill nulls with 0\n"
            "df[df['score'] > 80]   # filter rows\n"
            "df.groupby('dept')['salary'].mean()  # group & aggregate\n"
            "df.merge(df2, on='id') # join two dataframes\n"
            "```\n\n"
            "--- NUMPY (numerical computing) ---\n"
            "```python\n"
            "import numpy as np\n\n"
            "arr = np.array([1, 2, 3, 4, 5])\n"
            "print(arr.mean(), arr.std(), arr.shape)\n\n"
            "matrix = np.random.randn(3, 3)     # 3x3 random matrix\n"
            "np.dot(matrix, matrix.T)           # matrix multiplication\n"
            "```\n\n"
            "--- MATPLOTLIB / SEABORN (visualization) ---\n"
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "sns.histplot(df['score'], kde=True)\n"
            "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')\n"
            "sns.boxplot(x='dept', y='salary', data=df)\n"
            "plt.show()\n"
            "```\n\n"
            "--- SCIKIT-LEARN (machine learning) ---\n"
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import accuracy_score, classification_report\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "scaler = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test  = scaler.transform(X_test)\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X_train, y_train)\n"
            "print(accuracy_score(y_test, model.predict(X_test)))\n"
            "```"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 15. DATA STRUCTURES & ALGORITHMS (CS)
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["data structure", "linked list", "stack", "queue", "tree",
                               "binary tree", "graph", "hash map", "array", "sorting",
                               "binary search", "recursion", "dynamic programming",
                               "big o", "time complexity", "space complexity", "bfs", "dfs"]):
        return (
            "**Data Structures & Algorithms — CS Fundamentals:**\n\n"
            "--- DATA STRUCTURES ---\n"
            "- **Array/List** — O(1) access, O(n) insert/delete\n"
            "- **Linked List** — O(1) insert at head, O(n) search\n"
            "- **Stack (LIFO)** — push/pop; used in recursion, undo\n"
            "- **Queue (FIFO)** — enqueue/dequeue; used in BFS, scheduling\n"
            "- **Hash Map (Dict)** — O(1) average lookup, insert, delete\n"
            "- **Binary Search Tree** — O(log n) search if balanced\n"
            "- **Heap** — O(1) max/min access; used in priority queues\n"
            "- **Graph** — nodes + edges; directed/undirected, weighted\n"
            "- **Trie** — prefix tree; used in autocomplete, spellcheck\n\n"
            "--- SORTING ALGORITHMS ---\n"
            "| Algorithm | Best | Avg | Worst | Stable? |\n"
            "|---|---|---|---|---|\n"
            "| Bubble Sort | O(n) | O(n²) | O(n²) | Yes |\n"
            "| Selection Sort | O(n²) | O(n²) | O(n²) | No |\n"
            "| Insertion Sort | O(n) | O(n²) | O(n²) | Yes |\n"
            "| Merge Sort | O(n log n) | O(n log n) | O(n log n) | Yes |\n"
            "| Quick Sort | O(n log n) | O(n log n) | O(n²) | No |\n"
            "| Heap Sort | O(n log n) | O(n log n) | O(n log n) | No |\n\n"
            "--- SEARCHING ---\n"
            "- **Linear Search** — O(n); works on unsorted\n"
            "- **Binary Search** — O(log n); requires sorted array\n\n"
            "--- GRAPH ALGORITHMS ---\n"
            "- **BFS** — Level-by-level; shortest path in unweighted graph\n"
            "- **DFS** — Depth-first; connected components, topological sort\n"
            "- **Dijkstra** — Shortest path in weighted graph\n"
            "- **A*** — Heuristic shortest path (used in maps, games)\n\n"
            "--- DYNAMIC PROGRAMMING ---\n"
            "```python\n"
            "# Fibonacci with memoization (DP)\n"
            "def fib(n, memo={}):\n"
            "    if n in memo: return memo[n]\n"
            "    if n <= 1: return n\n"
            "    memo[n] = fib(n-1) + fib(n-2)\n"
            "    return memo[n]\n"
            "```\n\n"
            "**Big-O Cheat Sheet:**\n"
            "O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)\n\n"
            "**Practice:** LeetCode, HackerRank, GeeksforGeeks"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 16. INTERVIEW QUESTIONS
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["interview", "interview question", "interview prep", "placement",
                               "viva", "exam question", "quiz", "mcq", "job", "hire",
                               "resume", "career", "data scientist role", "ml engineer"]):
        return (
            f"**Data Science & ML Interview Questions — {level} Level:**\n\n"
            "--- STATISTICS (must-know) ---\n"
            "Q: What is the difference between mean, median, and mode?\n"
            "A: Mean=average; Median=middle value (robust to outliers); Mode=most frequent.\n\n"
            "Q: What is p-value? When do we reject H₀?\n"
            "A: Probability of observing data if H₀ is true. Reject H₀ when p < 0.05.\n\n"
            "Q: What is the Central Limit Theorem?\n"
            "A: The distribution of sample means approaches Normal as sample size increases (n≥30).\n\n"
            "--- MACHINE LEARNING ---\n"
            "Q: What is overfitting? How do you prevent it?\n"
            "A: Model performs well on training but poorly on test data.\n"
            "   Fix: Cross-validation, Dropout, Regularization (L1/L2), more data, pruning.\n\n"
            "Q: Explain Bias-Variance Tradeoff.\n"
            "A: High bias = underfitting (too simple); High variance = overfitting (too complex).\n"
            "   Goal: find the sweet spot with low bias AND low variance.\n\n"
            "Q: What is cross-validation? Why is it better than train/test split?\n"
            "A: K-Fold CV trains on K-1 folds, tests on 1; repeats K times.\n"
            "   Gives more reliable performance estimate — every sample is tested once.\n\n"
            "Q: What is the difference between precision and recall?\n"
            "A: Precision = TP/(TP+FP) — of all predicted positives, how many are correct.\n"
            "   Recall = TP/(TP+FN) — of all actual positives, how many did we catch.\n"
            "   F1 = harmonic mean of precision and recall.\n\n"
            "--- SQL ---\n"
            "Q: What is the difference between WHERE and HAVING?\n"
            "A: WHERE filters rows before aggregation; HAVING filters groups after GROUP BY.\n\n"
            "Q: What are window functions? Give an example.\n"
            "A: Perform calculations over a set of rows related to the current row.\n"
            "   Example: ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)\n\n"
            "--- PYTHON ---\n"
            "Q: What is the difference between list and tuple in Python?\n"
            "A: List is mutable (can change); Tuple is immutable (cannot change). Tuples are faster.\n\n"
            "Q: What does iloc vs loc do in pandas?\n"
            "A: iloc = integer-based indexing (position); loc = label-based indexing (name).\n\n"
            "--- NLP / LLM ---\n"
            "Q: What is the difference between stemming and lemmatization?\n"
            "A: Stemming crudely chops word endings; Lemmatization returns the proper base form.\n\n"
            "Q: What is the Transformer architecture?\n"
            "A: Neural network using self-attention to process all tokens in parallel.\n"
            "   Basis for BERT, GPT, IBM Granite.\n\n"
            "--- BEHAVIOURAL ---\n"
            "Q: Tell me about a data project you worked on.\n"
            "Q: How do you explain a complex ML model to a non-technical manager?\n"
            "Q: How do you handle missing/dirty data?\n\n"
            "**Tip:** STAR format for behavioural questions — Situation, Task, Action, Result!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 17. AI — Artificial Intelligence overview
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["artificial intelligence", "what is ai", "ai vs ml",
                               "types of ai", "narrow ai", "general ai", "agi",
                               "generative ai", "computer vision", "speech recognition",
                               "expert system", "robotics ai"]):
        return (
            "**Artificial Intelligence (AI) — Small to Large:**\n\n"
            "--- WHAT IS AI? ---\n"
            "AI = machines that simulate human intelligence (learning, reasoning, problem-solving).\n\n"
            "--- AI vs ML vs DL (hierarchy) ---\n"
            "```\n"
            "AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  LLMs\n"
            "```\n"
            "- **AI** — broad field; anything that makes machines 'smart'\n"
            "- **ML** — subset; machines learn from data\n"
            "- **DL** — subset of ML; neural networks with many layers\n"
            "- **LLMs** — subset of DL; trained on language\n\n"
            "--- TYPES OF AI ---\n"
            "- **Narrow AI (ANI)** — does ONE specific task well (e.g., spam filter, chess)\n"
            "- **General AI (AGI)** — human-level intelligence across all tasks (not yet achieved)\n"
            "- **Super AI (ASI)** — surpasses human intelligence (theoretical)\n\n"
            "--- AI SUBFIELDS ---\n"
            "- **Machine Learning** — learn patterns from data\n"
            "- **Natural Language Processing (NLP)** — understand/generate text\n"
            "- **Computer Vision (CV)** — understand images/video\n"
            "- **Speech Recognition** — convert speech to text (Siri, Alexa)\n"
            "- **Robotics** — physical AI agents\n"
            "- **Expert Systems** — rule-based AI (if-then logic)\n"
            "- **Generative AI** — create new content (text, images, code)\n\n"
            "--- GENERATIVE AI ---\n"
            "- **LLMs:** GPT-4, IBM Granite, LLaMA — generate text\n"
            "- **Image Gen:** DALL-E, Stable Diffusion, Midjourney\n"
            "- **Code Gen:** GitHub Copilot, IBM Granite Code\n"
            "- **Music Gen:** MusicLM, Suno AI\n\n"
            "--- AI ETHICS ---\n"
            "- Bias & Fairness — AI can inherit data biases\n"
            "- Explainability (XAI) — can we explain why the model decided X?\n"
            "- Privacy — training on personal data\n"
            "- Hallucination — LLMs confidently stating wrong facts\n"
            "- IBM is a leader in **Trustworthy AI** through watsonx.governance"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 18. DATA VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["visuali", "plot", "chart", "graph", "eda",
                               "exploratory data", "tableau", "dashboard", "plotly",
                               "seaborn", "heatmap", "histogram", "scatter", "box plot"]):
        return (
            "**Data Visualization — Small to Large:**\n\n"
            "--- WHY VISUALIZE? ---\n"
            "Human brains process visuals 60,000x faster than text. "
            "Good charts reveal patterns, outliers, and relationships instantly.\n\n"
            "--- CHART SELECTION GUIDE ---\n"
            "| Goal | Best Chart |\n"
            "|---|---|\n"
            "| Compare categories | Bar / Column Chart |\n"
            "| Show trend over time | Line Chart |\n"
            "| Part of a whole | Pie / Donut Chart |\n"
            "| Distribution | Histogram / KDE |\n"
            "| Relationship | Scatter Plot |\n"
            "| Spread & outliers | Box Plot / Violin |\n"
            "| Correlation matrix | Heatmap |\n"
            "| Geographical data | Choropleth Map |\n"
            "| Hierarchy | Treemap / Sunburst |\n\n"
            "--- PYTHON EXAMPLES ---\n"
            "```python\n"
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n\n"
            "# 1. Distribution\n"
            "sns.histplot(df['age'], kde=True, bins=30)\n\n"
            "# 2. Relationship\n"
            "sns.scatterplot(x='experience', y='salary', hue='dept', data=df)\n\n"
            "# 3. Correlation heatmap\n"
            "sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn')\n\n"
            "# 4. Box plot\n"
            "sns.boxplot(x='dept', y='salary', data=df)\n"
            "plt.xticks(rotation=45)\n"
            "plt.show()\n"
            "```\n\n"
            "--- EDA CHECKLIST ---\n"
            "1. df.shape, df.info(), df.describe()\n"
            "2. Missing values: df.isnull().sum()\n"
            "3. Distribution of each numeric column\n"
            "4. Correlation heatmap\n"
            "5. Outlier detection (box plots, Z-score)\n"
            "6. Target variable distribution\n\n"
            "--- TOOLS ---\n"
            "- **Python:** matplotlib, seaborn, plotly, bokeh\n"
            "- **Power BI** — interactive enterprise dashboards\n"
            "- **Tableau** — drag-and-drop; industry standard\n"
            "- **Excel** — quick charts for small datasets"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 19. IBM GRANITE / WATSONX
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["ibm", "granite", "watsonx", "watson", "ibm cloud",
                               "ibm ai", "watson studio", "autoai", "prompt lab"]):
        return (
            "**IBM Granite & watsonx.ai — Explained:**\n\n"
            "**IBM Granite** is a family of open, enterprise-grade LLMs by IBM Research.\n"
            "LearnMate is powered by **ibm/granite-3-3-8b-instruct**!\n\n"
            "--- IBM GRANITE MODELS ---\n"
            "- **Granite 3.3 8B Instruct** — instruction-following; used in LearnMate\n"
            "- **Granite 20B Multilingual** — supports 10+ languages\n"
            "- **Granite Code** — specialized for code generation\n"
            "- **Granite Time Series** — forecasting model\n\n"
            "--- WATSONX.AI PLATFORM ---\n"
            "- **Prompt Lab** — experiment with LLM prompts visually\n"
            "- **AutoAI** — automatically build, optimize, and deploy ML models\n"
            "- **Watson Studio** — Jupyter notebooks + pipelines in cloud\n"
            "- **Watson Machine Learning** — deploy & serve models at scale\n"
            "- **watsonx.governance** — AI fairness, bias detection, explainability\n\n"
            "--- CALLING IBM GRANITE IN PYTHON ---\n"
            "```python\n"
            "from ibm_watsonx_ai import APIClient, Credentials\n"
            "from ibm_watsonx_ai.foundation_models import ModelInference\n"
            "from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params\n\n"
            "creds = Credentials(url='https://us-south.ml.cloud.ibm.com', api_key='YOUR_KEY')\n"
            "client = APIClient(creds, project_id='YOUR_PROJECT_ID')\n\n"
            "model = ModelInference(\n"
            "    model_id='ibm/granite-3-3-8b-instruct',\n"
            "    api_client=client,\n"
            "    project_id='YOUR_PROJECT_ID',\n"
            "    params={Params.MAX_NEW_TOKENS: 500}\n"
            ")\n"
            "response = model.generate_text(prompt='Explain machine learning in simple terms:')\n"
            "print(response)\n"
            "```\n\n"
            "--- WHY IBM FOR STUDENTS? ---\n"
            "- IBM Cloud Lite = FREE (no credit card)\n"
            "- IBM SkillsBuild = Free courses + certifications\n"
            "- IBM certifications are industry-recognized\n"
            "- IBM Research publishes state-of-the-art AI papers\n\n"
            "Go to **IBM Granite Config** in the sidebar to enter your credentials!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 20. ROADMAP
    # ──────────────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["roadmap", "plan", "path", "curriculum", "syllabus",
                               "where to start", "how to start", "learning plan",
                               "study plan", "what to study"]):
        return (
            f"**Your {track} Learning Roadmap — Full Curriculum:**\n\n"
            "**PHASE 1 — Beginner (0-2 months)**\n"
            "1. Python Basics (variables, loops, functions, OOP)\n"
            "2. Statistics & Probability (mean, std, distributions, hypothesis testing)\n"
            "3. SQL Fundamentals (SELECT, JOIN, GROUP BY, subqueries)\n"
            "4. Excel for Data Analysis (pivot tables, VLOOKUP, charts)\n\n"
            "**PHASE 2 — Intermediate (2-5 months)**\n"
            "5. Python for DS (pandas, numpy, matplotlib, seaborn)\n"
            "6. Machine Learning with scikit-learn\n"
            "   - Supervised: Regression, Classification\n"
            "   - Unsupervised: Clustering, Dimensionality Reduction\n"
            "7. Data Visualization & EDA (Plotly, Power BI basics)\n"
            "8. Feature Engineering & Model Evaluation\n\n"
            "**PHASE 3 — Advanced (5-9 months)**\n"
            "9. Deep Learning (ANN, CNN, RNN with TensorFlow/PyTorch)\n"
            "10. NLP & Text Analysis (NLTK, spaCy, HuggingFace)\n"
            "11. LLMs & Prompt Engineering (GPT, IBM Granite, RAG)\n"
            "12. ML Deployment (Flask, FastAPI, Docker, Cloud)\n"
            "13. IBM watsonx.ai & Watson Studio\n\n"
            "**PHASE 4 — Portfolio & Career**\n"
            "14. Build 3-5 end-to-end projects on GitHub\n"
            "15. Compete on Kaggle (get a Bronze/Silver medal)\n"
            "16. Prepare interview answers (STAR format)\n"
            "17. Apply for internships / entry-level DS roles\n\n"
            f"At your **{level}** level, you're in Phase 2. Keep going — you're doing great!"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 21. DEFAULT — list all topics
    # ──────────────────────────────────────────────────────────────────────────
    return (
        f"Hi {name}! I'm your LearnMate AI Coach, powered by IBM Granite. "
        f"I can answer questions on a wide range of topics for your **{track}** journey.\n\n"
        "**Ask me about any of these:**\n\n"
        "**ML & AI:**\n"
        "- Types of ML (supervised, unsupervised, reinforcement)\n"
        "- ML Algorithms (Random Forest, SVM, KNN, XGBoost, etc.)\n"
        "- Classification vs Regression vs Clustering\n"
        "- Deep Learning & Neural Networks (CNN, RNN, LSTM, Transformers)\n"
        "- AI overview, Generative AI, Computer Vision\n\n"
        "**NLP & LLMs:**\n"
        "- NLP concepts (tokenization, TF-IDF, Word2Vec, sentiment analysis)\n"
        "- Large Language Models (BERT, GPT, IBM Granite, RAG, Prompt Engineering)\n\n"
        "**Programming:**\n"
        "- Python for Data Science (pandas, numpy, scikit-learn)\n"
        "- R Programming (ggplot2, dplyr, caret)\n"
        "- SQL (basics to advanced, window functions, CTEs)\n\n"
        "**Tools:**\n"
        "- Power BI (DAX, Power Query, dashboards)\n"
        "- Excel (VLOOKUP, pivot tables, VBA, Power Query)\n"
        "- IBM Granite & watsonx.ai\n\n"
        "**Foundations:**\n"
        "- Statistics & Probability (hypothesis testing, distributions)\n"
        "- Data Structures & Algorithms (CS fundamentals)\n"
        "- Data Visualization & EDA\n\n"
        "**Career & Language:**\n"
        "- Interview Questions (ML, SQL, Python, Stats, NLP)\n"
        "- English Communication for Data Scientists\n"
        "- Marathi Support (मराठीमध्ये मदत)\n\n"
        "*(Connect IBM Granite API Key in Config page for live personalized responses!)*"
    )


# ── Default student profile ────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "name": "Alex Student",
    "email": "alex@email.com",
    "country": "United States",
    "education": "Bachelor's in Computer Science",
    "goal": "Become a Data Scientist / ML Engineer at a top tech company",
    "track": "Data Science",
    "level": "Intermediate",
    "progress": 58,
    "streak": 14,
    "skills": {"Python": 80, "Statistics": 65, "Machine Learning": 55, "SQL": 70, "Data Visualization": 50, "Deep Learning": 30},
    "interests": ["Data Science", "Machine Learning", "Python", "Computer Science"],
    "courses_completed": 19,
    "hours_spent": 52,
    "badges": ["First Course 🥇", "7-Day Streak 🔥", "Fast Learner ⚡"],
}

def get_profile():
    return session.get("profile", DEFAULT_PROFILE.copy())

def save_profile(data):
    session["profile"] = data


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    profile = get_profile()
    last_course = session.get("last_course")
    assessment_score = session.get("assessment_score")
    return render_template("dashboard.html", profile=profile, page="dashboard",
                           last_course=last_course, assessment_score=assessment_score)


@app.route("/profile", methods=["GET", "POST"])
def profile_page():
    profile = get_profile()
    if request.method == "POST":
        profile["name"]      = request.form.get("name", profile["name"])
        profile["email"]     = request.form.get("email", profile["email"])
        profile["country"]   = request.form.get("country", profile["country"])
        profile["education"] = request.form.get("education", profile["education"])
        profile["goal"]      = request.form.get("goal", profile["goal"])
        profile["track"]     = request.form.get("track", profile["track"])
        interests = request.form.getlist("interests")
        if interests:
            profile["interests"] = interests
        save_profile(profile)
        return redirect(url_for("profile_page"))
    return render_template("profile.html", profile=profile, page="profile")


@app.route("/assessment")
def assessment():
    profile = get_profile()
    return render_template("assessment.html", profile=profile, page="assessment")


@app.route("/recommendations")
def recommendations():
    profile = get_profile()
    return render_template("recommendations.html", profile=profile, page="recommendations")


@app.route("/roadmap")
def roadmap():
    profile = get_profile()
    return render_template("roadmap.html", profile=profile, page="roadmap")


@app.route("/progress")
def progress():
    profile = get_profile()
    milestones = session.get("milestones", [])
    finished_courses = session.get("finished_courses", [])
    return render_template("progress.html", profile=profile, page="progress",
                           milestones=milestones, finished_courses=finished_courses)


@app.route("/chat")
def chat():
    profile = get_profile()
    ibm_configured = bool(
        session.get("ibm_api_key") or os.getenv("IBM_API_KEY")
    ) and bool(
        session.get("ibm_project_id") or os.getenv("IBM_PROJECT_ID")
    )
    return render_template("chat.html", profile=profile, page="chat", ibm_configured=ibm_configured)


@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        session["ibm_api_key"]    = request.form.get("api_key", "")
        session["ibm_project_id"] = request.form.get("project_id", "")
        session["ibm_model"]      = request.form.get("model", "ibm/granite-3-3-8b-instruct")
        session["ibm_url"]        = request.form.get("url", "https://us-south.ml.cloud.ibm.com")
        return redirect(url_for("config"))

    ibm_configured = bool(session.get("ibm_api_key") or os.getenv("IBM_API_KEY"))
    return render_template(
        "config.html",
        page="config",
        ibm_configured=ibm_configured,
        current_model=session.get("ibm_model", os.getenv("IBM_MODEL", "ibm/granite-3-3-8b-instruct")),
        current_url=session.get("ibm_url", os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")),
    )


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "message required"}), 400
    profile  = get_profile()
    response = ask_granite(data["message"], profile)
    return jsonify({"response": response})


@app.route("/api/config/test", methods=["POST"])
def api_test_config():
    data = request.get_json() or {}
    api_key    = data.get("api_key")    or session.get("ibm_api_key")    or os.getenv("IBM_API_KEY", "")
    project_id = data.get("project_id") or session.get("ibm_project_id") or os.getenv("IBM_PROJECT_ID", "")

    if not api_key or not project_id:
        return jsonify({"ok": False, "message": "API Key and Project ID are required."})

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        url = data.get("url", "https://us-south.ml.cloud.ibm.com")
        credentials = Credentials(url=url, api_key=api_key)
        client = APIClient(credentials, project_id=project_id)
        models = client.foundation_models.get_model_specs()
        return jsonify({"ok": True, "message": "Connected to IBM Cloud successfully! IBM Granite is ready."})
    except Exception as e:
        return jsonify({"ok": False, "message": f"Connection failed: {str(e)}"})


@app.route("/api/profile", methods=["POST"])
def api_profile():
    data = request.get_json() or {}
    profile = get_profile()
    profile.update({k: v for k, v in data.items() if k in DEFAULT_PROFILE})
    save_profile(profile)
    return jsonify({"ok": True})


@app.route("/api/complete-lesson", methods=["POST"])
def api_complete_lesson():
    """Mark a lesson as completed, update profile stats, award XP."""
    data = request.get_json() or {}
    course_id  = data.get("course_id", "")
    lesson_idx = data.get("lesson_idx", 0)
    course_title = data.get("course_title", "")
    lesson_title = data.get("lesson_title", "")

    profile = get_profile()

    # ── track completed lessons ─────────────────────────────────────────────
    completed = session.get("completed_lessons", {})
    key = f"{course_id}:{lesson_idx}"
    already_done = key in completed
    if not already_done:
        completed[key] = {"course_id": course_id, "lesson_idx": lesson_idx,
                          "course_title": course_title, "lesson_title": lesson_title}
        session["completed_lessons"] = completed

        # Update hours_spent (+0.5h per lesson)
        profile["hours_spent"] = profile.get("hours_spent", 0) + 1

        # Award XP → map lesson topics to skill improvements
        skill_map = {
            "python": "Python", "pandas": "Python", "numpy": "Python",
            "sklearn": "Machine Learning", "machine learning": "Machine Learning",
            "neural": "Deep Learning", "deep learning": "Deep Learning",
            "sql": "SQL", "database": "SQL", "query": "SQL",
            "statistics": "Statistics", "probability": "Statistics",
            "visualization": "Data Visualization", "seaborn": "Data Visualization",
            "matplotlib": "Data Visualization", "nlp": "Machine Learning",
        }
        title_lower = (course_title + " " + lesson_title).lower()
        for keyword, skill in skill_map.items():
            if keyword in title_lower and skill in profile["skills"]:
                current = profile["skills"][skill]
                profile["skills"][skill] = min(100, current + 2)
                break

    # ── track completed courses ─────────────────────────────────────────────
    # Count total lessons for this course
    total_lessons = data.get("total_lessons", 5)
    course_lessons_done = sum(
        1 for k in completed if k.startswith(f"{course_id}:")
    )
    course_finished = course_lessons_done >= total_lessons
    new_course = False

    if course_finished:
        finished_courses = session.get("finished_courses", [])
        if course_id not in finished_courses:
            finished_courses.append(course_id)
            session["finished_courses"] = finished_courses
            profile["courses_completed"] = profile.get("courses_completed", 0) + 1
            new_course = True
            # Add milestone
            milestones = session.get("milestones", [])
            milestones.insert(0, {"text": f"Completed: {course_title}", "xp": 500})
            session["milestones"] = milestones[:10]
            # Overall progress bump
            profile["progress"] = min(100, profile.get("progress", 0) + 3)
            # Badge for first course
            if profile["courses_completed"] == 1 and "First Course 🥇" not in profile.get("badges", []):
                profile.setdefault("badges", []).append("First Course 🥇")

    # ── last played ─────────────────────────────────────────────────────────
    session["last_course"] = {"id": course_id, "title": course_title,
                               "lesson": lesson_title, "progress": round(course_lessons_done / total_lessons * 100)}

    save_profile(profile)
    return jsonify({
        "ok": True,
        "already_done": already_done,
        "course_finished": course_finished,
        "new_course": new_course,
        "lessons_done": course_lessons_done,
        "total_lessons": total_lessons,
        "xp_awarded": 0 if already_done else 100,
        "profile": {
            "hours_spent": profile["hours_spent"],
            "courses_completed": profile["courses_completed"],
            "skills": profile["skills"],
            "progress": profile["progress"],
        }
    })


@app.route("/api/assessment-result", methods=["POST"])
def api_assessment_result():
    """Save assessment score and update skill levels accordingly."""
    data = request.get_json() or {}
    score   = data.get("score", 0)     # 0–100
    answers = data.get("answers", {})  # {topic: correct:bool}

    profile = get_profile()

    # Determine level based on score
    if score >= 80:
        profile["level"] = "Advanced"
        profile["progress"] = max(profile.get("progress", 0), 75)
    elif score >= 55:
        profile["level"] = "Intermediate"
        profile["progress"] = max(profile.get("progress", 0), 45)
    else:
        profile["level"] = "Beginner"
        profile["progress"] = max(profile.get("progress", 0), 15)

    # Update individual skill levels from topic answers
    topic_skill_map = {
        "python":     "Python",
        "statistics": "Statistics",
        "ml":         "Machine Learning",
        "sql":        "SQL",
        "algorithms": "Machine Learning",
        "dl":         "Deep Learning",
    }
    for topic, correct in answers.items():
        skill = topic_skill_map.get(topic)
        if skill and skill in profile["skills"]:
            delta = +8 if correct else -3
            profile["skills"][skill] = max(5, min(100, profile["skills"][skill] + delta))

    # Save assessment score
    session["assessment_score"] = score
    save_profile(profile)
    return jsonify({"ok": True, "level": profile["level"], "progress": profile["progress"]})


@app.route("/api/session-data")
def api_session_data():
    """Return aggregated session data for dashboard/progress pages."""
    profile   = get_profile()
    completed = session.get("completed_lessons", {})
    milestones = session.get("milestones", [])
    last_course = session.get("last_course")
    finished_courses = session.get("finished_courses", [])

    return jsonify({
        "profile": profile,
        "completed_lessons_count": len(completed),
        "finished_courses": finished_courses,
        "milestones": milestones,
        "last_course": last_course,
        "assessment_score": session.get("assessment_score"),
    })


# ── Roadmap data: 7 roles × 3 tiers ──────────────────────────────────────────
ROADMAP_DATA = {
    "data-scientist": {
        "label": "Data Scientist", "emoji": "📊", "color": "#4f8ef7",
        "description": "Master the full data science pipeline: statistics, Python, ML, and deployment.",
        "beginner": [
            {"id":"ds_b1","title":"Python Programming Basics",            "hours":8,  "status":"done",   "desc":"Variables, loops, functions, OOP, file I/O. Foundation of all DS work."},
            {"id":"ds_b2","title":"Statistics & Probability Essentials",  "hours":6,  "status":"done",   "desc":"Mean, median, std, distributions, hypothesis testing, p-values."},
            {"id":"ds_b3","title":"Introduction to Data Analysis",        "hours":5,  "status":"done",   "desc":"Pandas, NumPy, EDA workflow, data cleaning and transformation."},
            {"id":"ds_b4","title":"SQL for Data Science",                 "hours":6,  "status":"done",   "desc":"SELECT, JOIN, GROUP BY, Window Functions, CTEs, subqueries."},
            {"id":"ds_b5","title":"Excel & Power BI Basics",              "hours":4,  "status":"done",   "desc":"Pivot tables, VLOOKUP, basic dashboards, data storytelling."},
        ],
        "intermediate": [
            {"id":"ds_i1","title":"Machine Learning with scikit-learn",    "hours":14,"status":"active", "desc":"Supervised/unsupervised, model evaluation, cross-validation, pipelines."},
            {"id":"ds_i2","title":"Data Visualization (Plotly & Seaborn)", "hours":8, "status":"active", "desc":"EDA charts, interactive dashboards, storytelling with visuals."},
            {"id":"ds_i3","title":"Linear Algebra for ML",                 "hours":10,"status":"active", "desc":"Vectors, matrices, eigenvalues — the math behind PCA and neural nets."},
            {"id":"ds_i4","title":"Feature Engineering & Selection",       "hours":9, "status":"todo",   "desc":"Encoding, scaling, feature creation, RFE, feature importance."},
            {"id":"ds_i5","title":"IBM watsonx.ai & Watson Studio",        "hours":8, "status":"todo",   "desc":"AutoAI, Prompt Lab, WML deployments, watsonx.governance."},
        ],
        "advanced": [
            {"id":"ds_a1","title":"Deep Learning (TensorFlow & Keras)",    "hours":16,"status":"locked", "desc":"ANNs, CNNs, RNNs, LSTMs, Transformers. Build production models."},
            {"id":"ds_a2","title":"Natural Language Processing (NLP)",     "hours":12,"status":"locked", "desc":"Tokenization, TF-IDF, Word2Vec, BERT, HuggingFace fine-tuning."},
            {"id":"ds_a3","title":"ML Deployment (Flask + Docker)",        "hours":10,"status":"locked", "desc":"REST APIs for models, Docker containers, CI/CD, cloud deployment."},
            {"id":"ds_a4","title":"Kaggle & Portfolio Projects",           "hours":20,"status":"locked", "desc":"End-to-end projects: compete on Kaggle, build GitHub portfolio."},
        ],
    },
    "data-analyst": {
        "label": "Data Analyst", "emoji": "📈", "color": "#00c9a7",
        "description": "Turn raw data into business insights using SQL, Excel, Power BI, and Python.",
        "beginner": [
            {"id":"da_b1","title":"Excel Fundamentals",                   "hours":5, "status":"done",   "desc":"Formulas, VLOOKUP, pivot tables, charts, conditional formatting."},
            {"id":"da_b2","title":"SQL Essentials",                       "hours":6, "status":"done",   "desc":"SELECT, JOIN, GROUP BY, aggregations, subqueries."},
            {"id":"da_b3","title":"Data Storytelling",                    "hours":4, "status":"done",   "desc":"Chart selection, visual communication, executive summaries."},
            {"id":"da_b4","title":"Statistics for Analysts",              "hours":5, "status":"done",   "desc":"Averages, variance, distributions, business KPIs."},
        ],
        "intermediate": [
            {"id":"da_i1","title":"Power BI Desktop",                     "hours":10,"status":"active", "desc":"Data model, DAX formulas, interactive visuals, slicers, reports."},
            {"id":"da_i2","title":"Advanced SQL & Window Functions",       "hours":8, "status":"active", "desc":"CTEs, RANK, LAG/LEAD, PIVOT, performance tuning."},
            {"id":"da_i3","title":"Python for Data Analysis",             "hours":10,"status":"todo",   "desc":"pandas, seaborn, matplotlib — automate Excel tasks with Python."},
            {"id":"da_i4","title":"Tableau Basics",                       "hours":7, "status":"todo",   "desc":"Calculated fields, dual-axis charts, dashboards, stories."},
        ],
        "advanced": [
            {"id":"da_a1","title":"Power BI Service & Governance",        "hours":8, "status":"locked", "desc":"Publish, share, RLS, workspaces, dataflows, PL-300 prep."},
            {"id":"da_a2","title":"A/B Testing & Experimentation",        "hours":9, "status":"locked", "desc":"Hypothesis testing, statistical significance, experiment design."},
            {"id":"da_a3","title":"Data Warehousing Concepts",            "hours":10,"status":"locked", "desc":"Star schema, dimensional modeling, Snowflake, BigQuery basics."},
        ],
    },
    "software-developer": {
        "label": "Software Developer", "emoji": "💻", "color": "#7c5cd8",
        "description": "Build robust full-stack applications using modern software engineering principles.",
        "beginner": [
            {"id":"sd_b1","title":"Python or Java Fundamentals",          "hours":10,"status":"done",   "desc":"Syntax, OOP, error handling, modules, unit testing."},
            {"id":"sd_b2","title":"Data Structures & Algorithms",         "hours":12,"status":"done",   "desc":"Arrays, linked lists, stacks, queues, trees, hash maps, Big-O."},
            {"id":"sd_b3","title":"Git & Version Control",                "hours":4, "status":"done",   "desc":"Branching, merging, pull requests, GitHub workflow."},
            {"id":"sd_b4","title":"HTML/CSS/JavaScript Basics",           "hours":8, "status":"done",   "desc":"DOM manipulation, events, fetch API, responsive layouts."},
        ],
        "intermediate": [
            {"id":"sd_i1","title":"Backend: Flask or FastAPI",            "hours":12,"status":"active", "desc":"REST APIs, JWT auth, database integration, middleware, testing."},
            {"id":"sd_i2","title":"Databases: SQL & NoSQL",               "hours":9, "status":"active", "desc":"PostgreSQL, MongoDB, ORM (SQLAlchemy), schema design."},
            {"id":"sd_i3","title":"React or Vue Frontend",                "hours":14,"status":"todo",   "desc":"Components, state management, hooks, routing, API calls."},
            {"id":"sd_i4","title":"Testing & Clean Code",                 "hours":8, "status":"todo",   "desc":"Unit/integration tests, TDD, SOLID principles, code reviews."},
        ],
        "advanced": [
            {"id":"sd_a1","title":"System Design",                        "hours":15,"status":"locked", "desc":"Scalability, load balancing, caching, microservices, event-driven."},
            {"id":"sd_a2","title":"Cloud Deployment (AWS/IBM Cloud)",     "hours":12,"status":"locked", "desc":"Containers, serverless, CI/CD pipelines, IaC with Terraform."},
            {"id":"sd_a3","title":"Performance & Security",               "hours":10,"status":"locked", "desc":"Profiling, OWASP Top 10, auth patterns, rate limiting."},
        ],
    },
    "ai-engineer": {
        "label": "AI Engineer", "emoji": "🤖", "color": "#f59e0b",
        "description": "Design and deploy AI systems using LLMs, deep learning, and IBM Granite/watsonx.",
        "beginner": [
            {"id":"ai_b1","title":"Python for AI",                        "hours":8, "status":"done",   "desc":"NumPy, pandas, Jupyter, virtual envs — the AI toolkit."},
            {"id":"ai_b2","title":"Math for ML",                          "hours":10,"status":"done",   "desc":"Linear algebra, calculus (gradients), probability — the ML math backbone."},
            {"id":"ai_b3","title":"ML Foundations with scikit-learn",     "hours":10,"status":"done",   "desc":"Supervised/unsupervised, model evaluation, pipelines."},
            {"id":"ai_b4","title":"IBM Cloud Lite & watsonx Intro",       "hours":4, "status":"done",   "desc":"Free IBM Cloud setup, watsonx.ai Prompt Lab, Granite models."},
        ],
        "intermediate": [
            {"id":"ai_i1","title":"Deep Learning (PyTorch / TensorFlow)", "hours":18,"status":"active", "desc":"ANNs, CNNs, RNNs, training loops, loss functions, GPU training."},
            {"id":"ai_i2","title":"NLP & Transformers",                   "hours":14,"status":"active", "desc":"BERT, GPT architecture, HuggingFace, tokenizers, fine-tuning."},
            {"id":"ai_i3","title":"IBM Granite Fine-Tuning & Prompt Eng", "hours":10,"status":"todo",   "desc":"LoRA/QLoRA on ibm-watsonx-ai SDK, few-shot prompting, RAG basics."},
            {"id":"ai_i4","title":"RAG Pipelines (LangChain)",            "hours":10,"status":"todo",   "desc":"Embeddings, vector stores (Chroma/Pinecone), retrieval chains."},
        ],
        "advanced": [
            {"id":"ai_a1","title":"LLM Production & MLOps",               "hours":14,"status":"locked", "desc":"Model serving (vLLM), Docker, Kubernetes, A/B evaluation, monitoring."},
            {"id":"ai_a2","title":"AI Agents & Tool Use",                  "hours":12,"status":"locked", "desc":"ReAct agents, function calling, MCP, multi-agent orchestration."},
            {"id":"ai_a3","title":"Responsible AI & IBM Governance",       "hours":8, "status":"locked", "desc":"Bias detection, explainability (LIME/SHAP), watsonx.governance."},
        ],
    },
    "cs-fundamentals": {
        "label": "CS Fundamentals", "emoji": "🔬", "color": "#ef4444",
        "description": "Master core Computer Science theory that underpins every software and data career.",
        "beginner": [
            {"id":"cs_b1","title":"Programming Basics (C / Python)",      "hours":10,"status":"done",   "desc":"Syntax, conditionals, loops, functions, scope, recursion."},
            {"id":"cs_b2","title":"Discrete Mathematics",                 "hours":8, "status":"done",   "desc":"Sets, logic, proof techniques, combinatorics, graph theory."},
            {"id":"cs_b3","title":"Computer Architecture & OS",           "hours":8, "status":"done",   "desc":"CPU, memory hierarchy, processes, threads, scheduling, file systems."},
            {"id":"cs_b4","title":"Networking Basics",                    "hours":5, "status":"done",   "desc":"TCP/IP, HTTP/HTTPS, DNS, sockets, REST, OSI model."},
        ],
        "intermediate": [
            {"id":"cs_i1","title":"DSA (Algorithms & Complexity)",        "hours":16,"status":"active", "desc":"Sorting, searching, graphs (BFS/DFS/Dijkstra), DP, Big-O analysis."},
            {"id":"cs_i2","title":"Database Systems",                     "hours":10,"status":"active", "desc":"Relational model, indexing, transactions, ACID, normalization."},
            {"id":"cs_i3","title":"Object-Oriented Design Patterns",      "hours":8, "status":"todo",   "desc":"SOLID, Gang of Four patterns, UML, dependency injection."},
            {"id":"cs_i4","title":"Theory of Computation",                "hours":8, "status":"todo",   "desc":"Automata, regular languages, context-free grammars, Turing machines."},
        ],
        "advanced": [
            {"id":"cs_a1","title":"Compilers & Interpreters",             "hours":14,"status":"locked", "desc":"Lexing, parsing (AST), semantic analysis, code generation."},
            {"id":"cs_a2","title":"Distributed Systems",                  "hours":14,"status":"locked", "desc":"CAP theorem, consensus (Raft/Paxos), fault tolerance, replication."},
            {"id":"cs_a3","title":"Advanced Algorithms (NP, Approx.)",    "hours":12,"status":"locked", "desc":"NP-completeness, approximation algorithms, randomized algorithms."},
        ],
    },
    "devops-engineer": {
        "label": "DevOps Engineer", "emoji": "⚙️", "color": "#06b6d4",
        "description": "Automate, deploy, and monitor software systems at scale with modern DevOps practices.",
        "beginner": [
            {"id":"dv_b1","title":"Linux & Shell Scripting",              "hours":8, "status":"done",   "desc":"Bash, file system, permissions, cron, pipes, sed/awk."},
            {"id":"dv_b2","title":"Git & GitHub Workflows",               "hours":5, "status":"done",   "desc":"Branches, rebasing, GitHub Actions, PR workflows, gitops."},
            {"id":"dv_b3","title":"Networking for DevOps",                "hours":6, "status":"done",   "desc":"Firewalls, load balancers, DNS, VPC, subnets, security groups."},
            {"id":"dv_b4","title":"Python / Go Scripting",                "hours":6, "status":"active", "desc":"Automation scripts, API calls, config management basics."},
        ],
        "intermediate": [
            {"id":"dv_i1","title":"Docker & Containerization",            "hours":10,"status":"active", "desc":"Dockerfiles, images, networks, volumes, docker-compose, best practices."},
            {"id":"dv_i2","title":"Kubernetes Fundamentals",              "hours":14,"status":"todo",   "desc":"Pods, deployments, services, config maps, ingress, Helm charts."},
            {"id":"dv_i3","title":"CI/CD Pipelines",                      "hours":10,"status":"todo",   "desc":"GitHub Actions, Jenkins, ArgoCD, automated testing, deployment strategies."},
            {"id":"dv_i4","title":"Infrastructure as Code (Terraform)",   "hours":10,"status":"todo",   "desc":"HCL syntax, providers, state management, modules, IBM Cloud/AWS."},
        ],
        "advanced": [
            {"id":"dv_a1","title":"Monitoring & Observability",           "hours":10,"status":"locked", "desc":"Prometheus, Grafana, ELK stack, distributed tracing (Jaeger), alerting."},
            {"id":"dv_a2","title":"Advanced Kubernetes & Service Mesh",   "hours":14,"status":"locked", "desc":"Istio, Envoy, autoscaling, stateful sets, multi-cluster management."},
            {"id":"dv_a3","title":"Security (DevSecOps)",                 "hours":10,"status":"locked", "desc":"SAST/DAST, secret management (Vault), container scanning, RBAC."},
        ],
    },
    "machine-learning": {
        "label": "ML Engineer", "emoji": "🧠", "color": "#a855f7",
        "description": "Specialize in building, tuning, and shipping machine learning models into production.",
        "beginner": [
            {"id":"ml_b1","title":"Python & Scientific Stack",            "hours":8, "status":"done",   "desc":"NumPy, pandas, Matplotlib — the ML data toolkit."},
            {"id":"ml_b2","title":"Probability & Statistics",             "hours":8, "status":"done",   "desc":"Distributions, Bayes theorem, MLE, hypothesis testing."},
            {"id":"ml_b3","title":"Math for ML",                          "hours":10,"status":"done",   "desc":"Linear algebra, multivariate calculus, optimization (gradient descent)."},
            {"id":"ml_b4","title":"Intro ML with scikit-learn",           "hours":10,"status":"done",   "desc":"Regression, classification, clustering, pipelines, metrics."},
        ],
        "intermediate": [
            {"id":"ml_i1","title":"Ensemble Methods & Boosting",          "hours":10,"status":"active", "desc":"Random Forest, XGBoost, LightGBM, stacking, blending."},
            {"id":"ml_i2","title":"Deep Learning Fundamentals",           "hours":16,"status":"active", "desc":"ANN, CNN, RNN — training, regularization, hyperparameter tuning."},
            {"id":"ml_i3","title":"Feature Engineering & Selection",      "hours":8, "status":"todo",   "desc":"Domain features, target encoding, SHAP importance, selection wrappers."},
            {"id":"ml_i4","title":"Experiment Tracking (MLflow/W&B)",     "hours":6, "status":"todo",   "desc":"Logging metrics, artifact management, model registry, reproducibility."},
        ],
        "advanced": [
            {"id":"ml_a1","title":"MLOps & Model Serving",                "hours":14,"status":"locked", "desc":"FastAPI serving, BentoML, Triton, batch vs real-time inference."},
            {"id":"ml_a2","title":"Hyperparameter Optimization (Optuna)", "hours":8, "status":"locked", "desc":"Bayesian search, pruning, multi-objective optimization."},
            {"id":"ml_a3","title":"ML on IBM watsonx / AutoAI",           "hours":8, "status":"locked", "desc":"AutoAI experiments, model pipelines, deployment on WML."},
        ],
    },
}


@app.route("/api/roadmap/<role>")
def api_roadmap(role):
    """Return full roadmap data for the requested role (no auth required)."""
    data = ROADMAP_DATA.get(role)
    if not data:
        return jsonify({"error": "Role not found", "available": list(ROADMAP_DATA.keys())}), 404
    return jsonify(data)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\nLearnMate (IBM Granite) is starting...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
