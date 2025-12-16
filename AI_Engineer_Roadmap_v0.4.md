# AI Engineer Roadmap

## How to Use This Roadmap

### Guidelines
- **Time commitment:** 5-7 hours per week
- **Total duration:** ~175 hours (~35 weeks at 5 hrs/week, ~25 weeks at 7 hrs/week)
- **Buffer:** 40% buffer already included in time estimates. If you miss a week, resume without guilt.
- **Tracking:** Copy this file, mark items with `[x]` and add completion date in parentheses

### Stay Updated (Subscribe Now)
- The Batch (Andrew Ng) — https://www.deeplearning.ai/the-batch/
- Anthropic Blog — https://www.anthropic.com/research
- OpenAI Blog — https://openai.com/blog/
- Simon Willison's Blog — https://simonwillison.net/

### Key Resources
- **Hugging Face** — https://huggingface.co/
  The central hub for AI/ML: model leaderboards, pre-trained models, datasets, tutorials, Spaces for demos
  Essential resource throughout your journey — bookmark it

### Communities (Join for Support)
- Hugging Face Discord — https://huggingface.co/join/discord
- LangChain Discord — https://discord.gg/langchain
- r/LocalLLaMA — https://reddit.com/r/LocalLLaMA
- r/LLM - https://www.reddit.com/r/LLM/
- r/ClaudeAI - https://www.reddit.com/r/ClaudeAI/
- r/PromptEngineering - https://www.reddit.com/r/PromptEngineering/
- r/GeminiAI - https://www.reddit.com/r/GeminiAI/


---

## Progress

**Name:** _______________  
**Started:** YYYY-MM-DD  

---

## Prerequisites

Before starting, ensure you have:
-  Programming experience (1+ years in any language)
-  Basic understanding of HTTP APIs and JSON
-  Git basics (commits, branches, pull requests)
-  English proficiency (B2+ level)

---

## Level 1: Conceptual Foundation (2w total)

### 1.1 What is AI/ML (~1 hr)
- [ ] [Difference Between Classical Programming and Machine Learning (20 min)](https://www.reddit.com/r/AskComputerScience/comments/18tb705/difference_between_classical_programming_and/)
- [ ] [IBM: What is Machine Learning? (8 min)](https://www.youtube.com/watch?v=9gGnTQTYNaE)
- [ ] [StatQuest: A Gentle Introduction to Machine Learning (13 min)](https://www.youtube.com/watch?v=Gv9_4yMHFhI)
- [ ] [Google's intro to ML: what is Machine Learning (10 min)](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml)
- [ ] [Optional] [Microsoft Learn: Introduction to machine learning concepts (1.5 hr)](https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/)


**Check Yourself:**
- [ ] How would you explain the difference between traditional programming and ML to a non-technical colleague?
- [ ] If someone says "AI", "ML", and "Deep Learning" — can you explain how these terms relate to each other?
- [ ] Given a business problem, can you identify whether it needs supervised, unsupervised, or reinforcement learning?
- [ ] Why can't you just write rules manually for recognizing handwritten digits?

### 1.2 Problem Types (~1.5 hrs)

- [ ] [Micrososft Learn: Introduction to AI concepts (30 min)](https://learn.microsoft.com/en-us/training/modules/get-started-ai-fundamentals/)

- [ ] [Infinite Codes: All Machine Learning algorithms explained (17 minues)](https://www.youtube.com/watch?v=E0Hmnixke2g). Mighe be a bit technical but gives a good overview along with use cases

- [ ] **Clustering**. Examples: customer segmentation, anomaly detection
      When to use: finding patterns without labels.
       [Computerphile: Data Analysis 7: Clustering (16 min)](https://www.youtube.com/watch?v=KtRLF6rAkyo)

- [ ] **Classification**. Examples: spam detection, sentiment analysis, image recognition. When to use: predicting categories/labels. 
[Computerphile: Data Analysis 8: Classifying Data (16 min)](https://www.youtube.com/watch?v=1opLSwlqBSI&list=PLzH6n4zXuckpfMu_4Ff8E7Z1behQks5ba)

- [ ] **Regression**. Examples: price prediction, demand forecasting, risk scoring. When to use: predicting continuous values. 
[Computerphile: Data Analysis 9: Data Regression (16 min)](https://www.youtube.com/watch?v=ek0yyL8iV7I&list=PLzH6n4zXuckpfMu_4Ff8E7Z1behQks5ba)

- [ ] **Generation**. Examples: text generation, image creation, code completion. When to use: creating new content. 
[Google: "Introduction to Generative AI" (22 min)](https://www.youtube.com/watch?v=G2fqAlgmoPo) - optional deep dive
[Optional] [Microsoft Learn: Introduction to generative AI and agents (37 min)](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/)

**Check Yourself:**
- [ ]A client wants to predict which customers will churn next month — what problem type is this?
- [ ] You need to group similar support tickets automatically without predefined categories — what approach?
- [ ] What's the key difference between classification and regression in terms of output?
- [ ] When would you choose clustering over classification?
- [ ] Can you name a real-world example for each problem type from your own domain/experience?

### 1.3 Inside a neural network (~1.5 hrs)
- [ ] **Neurons, layers, weights, biases**
      [3Blue1Brown: But what is a neural network (19 min)](https://www.3blue1brown.com/lessons/neural-networks)

- [ ] **Gradient descent intuition**
      [3Blue1Brown: "Gradient descent, how neural networks learn (21 min)"](https://www.3blue1brown.com/lessons/gradient-descent)
      [3Blue1Brown: Analyzing our neural network (10 min)](https://www.3blue1brown.com/lessons/neural-network-analysis)

- [ ] **Backpropagation**
      [3Blue1Brown: "What is backpropagation really doing?" (13 min)](https://www.3blue1brown.com/lessons/backpropagation)
      [Optional] [3Blue1Brown: Backpropagation calculus (10 min)](https://www.3blue1brown.com/lessons/backpropagation-calculus)

- [ ] **Training vs Inference**
      [IBM Technology: AI Inference: The Secret to AI's Superpowers (11 min)](https://www.youtube.com/watch?v=XtT5i0ZeHHE)

- [ ] **Bias and variance**
      [StatQuest: Machine Learning Fundamentals: Bias and Variance (6 min)
](https://www.youtube.com/watch?v=EuBBz3bI-aA)

- [ ] **Overfitting & underfitting**
      [IBM Technology: Overfitting, Underfitting, and Bad Data Are Ruining Your Predictive Models (7 min)](https://www.youtube.com/watch?v=0RT2Q0qwXSA)

**Check Yourself:**
- [ ] What is gradient descent trying to achieve? Explain it like you're describing walking down a hill.
- [ ] Why do we need backpropagation? What problem does it solve in training neural networks?
- [ ] What's the difference between training and inference? Which one is more expensive computationally, and why?
- [ ] What is bias in machine learning? What happens when a model has high bias?
- [ ] What is variance in machine learning? What happens when a model has high variance?
- [ ] How are bias and variance related to underfitting and overfitting?
- [ ] Your model performs great on training data (98% accuracy) but poorly on new data (60% accuracy) — what's the problem?
- [ ] Your model performs poorly on both training and test data — what's likely wrong?
- [ ] How can you detect if your model is overfitting during training?

### 1.4 AI Task Types with Business Examples (~2 hrs)

- [ ] **Overview: AI Applications in Business**
      [TED: How AI Could Empower Any Business | Andrew Ng (11 minutes)](https://www.youtube.com/watch?v=reUZRyXxUs4)

- [ ] **OCR (Optical Character Recognition)**
      [IBM Technology: "What is OCR?" (6 min)](https://www.youtube.com/watch?v=or8AcS6y1xg)
      **Use cases:** invoice processing, document digitization, license plates

- [ ] **Natural Language Processing (NLP)**
      [IBM Technology: "Natural Language Processing" (10 min)](https://www.youtube.com/watch?v=fLvJ8VdHLA0)
      [Optional] [Microsoft Learn: Introduction to natural language processing concepts
(30 min)](https://learn.microsoft.com/en-us/training/modules/introduction-language/)**Use cases:** extracting names/dates from contracts, parsing resumes

- [ ] **Sentiment Analysis**
      [IBM Technology: "What is Sentiment Analysis?" (10 min)](https://www.youtube.com/watch?v=5HQCNAsSO-)
      **Use cases:** customer feedback, social media monitoring

- [ ] **Recommendation Systems**
      [Nvidia: Recommendation system (20 min)](https://www.nvidia.com/en-us/glossary/recommendation-system/)
      [Optional] [CS50: Recommender Systems | Linus Torvalds (14 min)](https://www.youtube.com/watch?v=Eeg1DEeWUjA)
      **Use cases:** product suggestions, content personalization

- [ ] **Conversational AI**
      [Jeff Su: AI Agents, Clearly Explained (10 min)](https://www.youtube.com/watch?v=FwOTs4UxQS4)
      [Alternative] [IBM Technology: What are AI agents (13 min)](https://www.youtube.com/watch?v=F8NKVhkZZWI)
      **Use cases:** chatbots, virtual assistants, customer support

- [ ] **Computer Vision**
      [AltexSoft: How Computer Vision Applications Work (13 min)
](https://www.youtube.com/watch?v=oGvHtpJMO3M)[Alternative] [IBM: What is computer vision (20 min)](https://www.ibm.com/think/topics/computer-vision)
[Optional] [Microsoft Learn: Introduction to computer vision concepts (34 min)](https://learn.microsoft.com/en-us/training/modules/introduction-computer-vision/)
      **Use cases:** quality control, object detection, medical imaging

**Check Yourself:**
- [ ] A company wants to automatically extract vendor names, dates, and amounts from thousands of scanned invoices — which AI task types are involved?
- [ ] Your client wants to understand if customer reviews are positive or negative — what task type is this?
- [ ] An e-commerce site wants to show "customers also bought" — what AI approach would you recommend?
- [ ] A factory wants to detect defective products on a conveyor belt — what task type applies?


### 1.5 Introduction to LLMs (~1.5 hrs)
- [ ] Wha are LLMs?
      [3Blue1Brown: Large Language Models explained briefly (8 min)](https://www.3blue1brown.com/lessons/mini-llm)
      [Alternative] [IBM Technology: How Large Language Models Work (6 min)](https://www.youtube.com/watch?v=5sLYAQS9sWQ)

- [ ] **Transformers**
      [3Blue1Brown: Transformers, the tech behind LLMs (27 min)](https://www.3blue1brown.com/lessons/gpt)

- [ ] **Tokenization**
      [Matt Pocock: Most devs don't understand how LLM tokens work (11 min)](https://www.youtube.com/watch?v=nKSk_TiR8YA)


**Check Yourself:**
- How do LLMs work?
- What do transformers do?
- What is a token?

### Level 1 Checkpoint

After completing Level 1, you should be able to:

**Conceptual Understanding (1.1, 1.3)**
- [ ] Explain the difference between traditional programming and ML to a non-technical person
- [ ] Describe how a neural network "learns" without using the word "magic"
- [ ] Explain gradient descent using a simple analogy (e.g., walking downhill blindfolded)
- [ ] Distinguish between training and inference — and why this matters for costs

**Problem Framing (1.2)**
- [ ] Identify whether a business problem is classification, regression, clustering, or generation
- [ ] Explain when you'd use supervised vs unsupervised learning

**Model Quality (1.3)**
- [ ] Define overfitting and underfitting in plain terms
- [ ] Recognize symptoms: "great on training data, bad on new data" → overfitting
- [ ] Explain the bias-variance tradeoff at a high level

**Business Applications (1.4)**
- [ ] Map real business problems to AI task types (OCR, NER, sentiment, recommendations, etc.)
- [ ] Given a scenario like "extract data from invoices," identify which AI capabilities apply

**LLM Foundations (1.5)**
- [ ] Explain what makes LLMs different from classical ML models
- [ ] Understand what tokens are and why they matter (pricing, context limits)

---

## Level 2: Python for AI (~1w-2w total)

### 2.1 Python Refresher (~2 hrs)
> Note: This assumes programming experience. Focus on Python-specific patterns used heavily in AI/ML code.

- [ ] **Python for experienced developers (quick overview)**
      [Learn X in Y Minutes: Python](https://learnxinyminutes.com/docs/python/)
      (Fast syntax overview — 30 min read for experienced devs)
      [Alternative] [Python Official Tutorial](https://docs.python.org/3/tutorial/) — skim as reference

- [ ] **Iterables**
      [Bro Code: Learn Python iterables (6 min)](https://www.youtube.com/watch?v=VL_g3LjsFqs)
       (Processing batches of embeddings, streaming chunks from LLM responses, iterating through dataset rows)

- [ ] **Working with dictionaries**
      [Corey Schafer: Dictionaries (10 min)](https://www.youtube.com/watch?v=daefaLgNkw0)
      [Bro Code: learn Python generations (8 min)](https://www.youtube.com/watch?v=G1lJeEIl05o)
      **Key patterns:** nested access, `.get()` with defaults, dict comprehensions, merging dicts

- [ ] **List comprehensions, generators**
      [Bro Code: List comprehensions (10 minutes)](https://www.youtube.com/watch?v=YlY2g2xrl6Q)
       (Transforming datasets, filtering results, memory-efficient processing of large document collections)

- [ ] **Args, kwargs**
      [Bro Code: *args & **kwargs (15 min)](https://www.youtube.com/watch?v=Vh__2V2tXUM)
       (Wrapping API calls with flexible parameters, building configurable pipeline functions)


- [ ] **String manipulation**
      [Bro Code: string methods (12 min)](https://www.youtube.com/watch?v=tb6EYiHtcXU)
      **Key patterns:** f-strings, `.format()`, `.strip()`, `.split()`, `.join()`, regex basics
      (Building prompt templates, cleaning extracted text, parsing LLM outputs)


- [ ] **JSON serialization/deserialization**
      [Tech with Tim: How to use JSON in Python (6 min)](https://youtu.be/-51jxlQaxyA?si=FOZLi2UBk4G4BZMl)
      **Key operations:** `json.loads()`, `json.dumps()`, handling nested structures, custom encoders
      (Sending/receiving data from AI APIs, storing structured outputs, configuration files)


- [ ] **Context managers (`with` statements)**
      [2MinutesPy: What exactly are context managers (2 min)](https://www.youtube.com/watch?v=IQ20WLlEHbU)
      **Key patterns:** file handling, database connections, custom context managers
      (Ensuring files close properly, managing database sessions, handling API client connections)


- [ ] **File operations**
      [Bro Code: Read files (7 min)](https://www.youtube.com/watch?v=GWBWQnWNWBI)
      [Bro Code: Writing files (14 min)](https://www.youtube.com/watch?v=1IYrmTTKOoI)
      **Key patterns:** reading text/binary files, handling encodings, working with paths
      (Loading prompts from files, reading documents for RAG, saving model outputs)

- [ ] **HTTP requests**
      [Bro code: Request API data (9 min)](https://www.youtube.com/watch?v=JVQNywo4AbU)
      **Key patterns:** GET/POST, headers, JSON payloads, handling responses
      (Calling OpenAI/Anthropic APIs, fetching external data, webhook integrations)

- [ ] **Exception handling**
      [Bro Code: exception handling (6 min)](https://www.youtube.com/watch?v=V_NXT2-QIlE)
      **Key patterns:** try/except/finally, catching specific exceptions, raising custom exceptions
      (Handling API rate limits, managing timeouts, graceful fallbacks when LLM calls fail)

- [ ] **Decorators**
      [Bro Code: learn Python decorators (7 min)](https://www.youtube.com/watch?v=U-G-mSd4KAE)
      (Adding retry logic to API calls, logging function inputs/outputs, caching expensive operations)


- [ ] **Type hints**
      [Indently: type hints in Python explained (5 min)](https://www.youtube.com/watch?v=15WB30NqDT0)
      (Required by Pydantic for validation, improves IDE autocomplete, standard in FastAPI/LangChain)


**Check Yourself:**
- [ ] You have a list of 1000 document chunks and need to extract only those containing the word "revenue" — how would you do this in one line?
- [ ] Your API response is a nested dict: `{"data": {"user": {"name": "John"}}}`. How do you safely access the name without crashing if any key is missing?
- [ ] You're building a function that wraps an LLM API call, but different calls need different optional parameters (temperature, max_tokens, etc.). How would you design the function signature?
- [ ] You received a JSON string from an API. How do you convert it to a Python dict, modify a field, and convert it back to a JSON string?
- [ ] You need to read a 500MB log file to find error messages, but loading it all into memory crashes your script. What approach would you use?
- [ ] You're calling an external API that sometimes times out. How would you handle this gracefully and retry up to 3 times?
- [ ] Your prompt template needs to include a variable and a float formatted to 2 decimal places. Write the f-string.
- [ ] You notice you're adding the same retry logic to multiple API-calling functions. How could decorators help?
- [ ] Why does `with open(...)` matter when processing hundreds of files in a loop?


### 2.2 Development Environment (~1.5 hrs)

- [ ] **Jupyter Notebooks**
      [Corey Schafer: Jupyter Notebook Tutorial (30 min)](https://www.youtube.com/watch?v=HW29067qVWk)
      **Key concepts:** cells, markdown, running code, keyboard shortcuts
      (Interactive experimentation, data exploration, prototyping ML pipelines)
      **Alternatives:**
      - [Google Colab](https://colab.research.google.com/) — free, cloud-based, free GPU access
      - [Colab Getting Started Guide](https://colab.research.google.com/notebooks/intro.ipynb)
      - [Kaggle Notebooks](https://www.kaggle.com/code) — similar to Colab, integrated with datasets
      - VS Code with Jupyter extension — if you prefer staying in your IDE

- [ ] **Virtual environments & pip**
      [Corey Schafer: pip Tutorial (14 min)](https://www.youtube.com/watch?v=U2ZN104hIcc)
      [Corey Schafer: Python venv Tutorial (16 min)](https://www.youtube.com/watch?v=Kg1Yvry_Ydk)
      **Key patterns:** creating/activating venv, `pip install`, `requirements.txt`, `pip freeze`
      (Isolating project dependencies, reproducible environments, avoiding "works on my machine")
      **Conda alternative (optional):**
      [Corey Schafer: Anaconda Tutorial (25 min)](https://www.youtube.com/watch?v=YJC6ldI3hWk)
      (Popular in data science/ML — manages Python versions + packages together)

**Check Yourself:**
- [ ] You're experimenting with a new embedding model and want to see results immediately after each code change. What tool would you use?
- [ ]A colleague sends you their notebook but you don't want to install anything locally. How can you run it?
- [ ]You're starting a new AI project. How do you create an isolated environment so it doesn't conflict with your other projects?
- [ ] Your project needs `langchain`, `openai`, and `pandas`. How do you install them and save the list so others can replicate your setup?
- [ ] What command shows all packages installed in your current environment?
- [ ] Why would you use Colab over a local Jupyter notebook for training a model?

### 2.3 Data Libraries (~4-6 hrs)
> Note: The video tutorials below are comprehensive (~1 hr each). You don't need to watch them from beginning to end — use them as reference to check specific topics you'll be working with. Watch entirely only if you prefer a thorough walkthrough.

- [ ] **NumPy (arrays, operations)**
      [freeCodeCamp: NumPy for Beginners (1 hr)](https://www.youtube.com/watch?v=QUT1VHiLmmI)
      [Alternative] [Code bro: learn Numpy (1 hr)](https://www.youtube.com/watch?v=VXU4LSAQDSc)
      [Alternative] [Derek Bans: Numpy full course (1 hr)](https://www.youtube.com/watch?v=8Y0qQEh7dJg)
      **Key patterns:** creating arrays, indexing/slicing, reshaping, broadcasting, basic math operations
      (Embeddings are NumPy arrays, vector similarity calculations, batch processing)
      **Reference:** [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

- [ ] **Pandas (DataFrames, manipulation)**
      [Keith Galli: Complete Python Pandas Data Science Tutorial (1.5 hr)](https://www.youtube.com/watch?v=vmEHCJofslg)
      [Alternative] [Bro Code: learn Pandas (1 hr)](https://www.youtube.com/watch?v=VXtjG_GzO7Q)
      [Alternative] [Derek Banas: Pandas full course](https://www.youtube.com/watch?v=PcvsOaixUh8)
      **Key patterns:** loading CSV/JSON, filtering rows, selecting columns, groupby, handling missing values, merging datasets
      **Reference:** [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

- [ ] **Matplotlib (visualization basics)**
      [Corey Schafer: Matplotlib Tutorial (35 min)](https://www.youtube.com/watch?v=UO98lJQ3QGI)
      [Alternative] [Bro Code: start using Matplotlib (7 min)](https://www.youtube.com/watch?v=2KY5AaFvWtE&list=PLZPZq0r_RZONpa_Xg1MmyjmctJjL4l3Ay). The whole matplotlib playlist is 1 hr.
      [Alternative] [Derek Banas: Matplotlib full course (1.5 hr)](https://www.youtube.com/watch?v=wB9C0Mz9gSo)
      **Key patterns:** line plots, bar charts, histograms, scatter plots, subplots, saving figures

- [ ] **Scikit-learn (ML utilities)**
      [freeCodeCamp: Scikit-learn Tutorial (2 hrs)](https://www.youtube.com/watch?v=0B5eIE_1vpU) — watch first 30 min for basics
      **Key patterns:** preprocessing utilities, train/test split, basic model fitting

- [ ] **[Optional] Seaborn (statistical visualization)**
      [Derek Banas: Seaborn Tutorial (60 min)](https://www.youtube.com/watch?v=6GUZXDef2U0)
      **Key patterns:** heatmaps, distribution plots, correlation matrices, pair plots
      **When to use:** Matplotlib for basic/custom plots, Seaborn for statistical visualizations with less code

**Check Yourself:**
- [ ] You have two lists of 1000 numbers each and need to compute their element-wise product. How would you do this efficiently with NumPy?
- [ ] You receive embeddings from an API as a list of lists. How do you convert them to a NumPy array and check its shape?
- [ ] You have a CSV with 100,000 customer records. How do you load it and filter to only customers from "Germany" with purchases over $100?
- [ ] Your dataset has missing values in the "email" column. How do you find how many are missing and fill them with "unknown"?
- [ ] You want to visualize how model accuracy changed over 50 training epochs. What type of plot would you use and how?
- [ ] You need to show the correlation between 10 different features in your dataset. Which library makes this easiest?

### Level 2 Checkpoint

After completing Level 2, you should be able to:

**Python Fundamentals (2.1)**
- [ ] Write Pythonic code using list comprehensions, generators, and dict operations
- [ ] Handle API responses: parse JSON, access nested data safely, handle missing keys
- [ ] Read/write files using context managers
- [ ] Make HTTP requests with proper error handling (timeouts, retries)
- [ ] Use type hints to document function signatures
- [ ] Apply decorators for cross-cutting concerns (logging, retries)

**Development Environment (2.2)**
- [ ] Set up and activate a virtual environment for a new project
- [ ] Install packages and generate requirements.txt for reproducibility
- [ ] Work comfortably in Jupyter notebooks (or Colab) for experimentation
- [ ] Know when to use notebooks vs regular Python scripts

**Data Libraries (2.3)**
- [ ] Create, reshape, and perform operations on NumPy arrays
- [ ] Load, filter, group, and transform data with Pandas
- [ ] Handle missing values and merge datasets
- [ ] Create basic visualizations (line plots, histograms, heatmaps) with Matplotlib/Seaborn

**Practical Integration**
- [ ] Given a JSON file with messy data, load it into Pandas, clean it, and visualize key distributions
- [ ] Set up a fresh project environment and install dependencies from a requirements file

---

## Level 3: Data & Preprocessing (~2w-3w total)

> **Why this matters:** 80% of AI work is data preparation. Poor data = poor results.

### 3.1 Data Quality (~2-3 hrs)

- [ ] **Data cleaning, filtering, handling missing data, duplicate detection**
      [Keith Galli: Cleaning data in Pandas (30 min)](https://www.youtube.com/watch?v=KdmPHEnPJPs)
      [Bro Code: learn Pandas in 1 hour, data cleaning (51:00 - 1:00:00) (10 min)](https://youtu.be/VXtjG_GzO7Q?si=pEE_E8fZ0Y6qO52R&t=3060)
      **Key patterns:** detecting nulls (`isna()`), `duplicated()`, `drop_duplicates()`, dropping vs imputing, fillna strategies (mean, median, forward-fill)
      (Missing data in training sets degrades model quality; in RAG, missing fields break downstream processing)

      **Reference:** [Pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

- [ ] **Outlier detection and handling**
      [Absent data: find outliers in python - 4 simple ways (12 min)](https://www.youtube.com/watch?v=NEuU1gaUyww)
      **Key patterns:** IQR method, Z-score, visual detection (box plots), when to remove vs cap vs keep
      (Outliers in embeddings or training data can distort model behavior)

      **Reference:** [Scikit-learn: Outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

- [ ] **Data normalization/standardization**
      [Geekforgeeks: data noramization with Pandas (10 min)](https://www.geeksforgeeks.org/python/data-normalization-with-pandas/)
      **Key patterns:** StandardScaler (z-score), MinMaxScaler (0-1 range), when to use which
      (Many ML algorithms require scaled features; embeddings are typically normalized for similarity search)
      
      Reference: [Scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

- [ ] **Data validation**
      [pixegami: Pydantic tutorial (11 min)](https://www.youtube.com/watch?v=XIdQ6gO3Anc)
      **Key patterns:** type checking, range validation, schema enforcement
      (Catching bad data early prevents silent failures in AI pipelines)

**Check Yourself:**
- [ ] Your dataset has 15% missing values in the "income" column. What are your options and how do you decide which approach to use?
- [ ] You notice some rows appear multiple times in your training data. Why is this a problem and how do you fix it?
- [ ] Your data has ages ranging from 0 to 500 — some are clearly errors. How do you detect and handle these outliers?
- [ ] You're preparing features for a model and one column ranges 0-1 while another ranges 0-1,000,000. Why is this a problem?
- [ ] What's the difference between normalization and standardization? When would you choose one over the other?
- [ ] You're building a pipeline that ingests JSON from an external API. How do you ensure the data matches your expected format before processing?


### 3.2 Working with Unstructured Data (~5.5 hrs)

> Most business data lives in documents, not databases. Extracting and cleaning text from PDFs, Word docs, and other formats is essential for AI applications.

> **Note:** The PDF and Word extraction videos are reference material — you don't need to memorize library APIs. In practice, you'll use LLMs to help write extraction code. Focus on understanding: (1) what challenges exist (scanned vs native PDFs, tables, formatting), (2) which libraries are available, and (3) the theory behind text preprocessing and document splitting.

- [ ] **Text extraction from PDFs**
      [NeuralNine: Extracting Text from PDF Files in Python (13 min)](https://www.youtube.com/watch?v=w2r2Bg42UPY)
      **Libraries:** `pdfminer.six`, `PyMuPDF`,  `pdfplumber`, `PyPDF2`
      (Invoices, contracts, reports, manuals — most enterprise documents are PDFs)

- [ ] **Text extraction from Word documents**
      [NeuralNine: Word file processing in Python (20 min)](https://www.youtube.com/watch?v=so2illANiRw)
      **Library:** `python-docx`
      **Key patterns:** extracting paragraphs, tables, handling formatting
      (HR policies, internal documentation, templates — common in enterprise settings)

- [ ] **Image preprocessing basics (reference)**
      [Pillow: tutorial (20 min)](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)
      **Library:** `Pillow (PIL)`
      **Key patterns:** loading/saving images, resizing, format conversion, basic transformations
      (Preparing images for vision models, thumbnail generation, format standardization)
      
      Reference: [Pillow Documentation](https://pillow.readthedocs.io/)

- [ ] **Web scraping basics**
      [BeautifulSoup + Requests | Web Scraping in Python (7 min)](https://www.youtube.com/watch?v=bargNl2WeN4)
      **Libraries:** `requests`, `BeautifulSoup`
      **Key patterns:** fetching pages, parsing HTML, extracting text and links
      (Scraping documentation, articles, product pages — common data source for AI applications)

- [ ] **Text cleaning and preprocessing**
      [Geekforgeeks: Text preprocessing in Python (10 min)](https://www.geeksforgeeks.org/machine-learning/text-preprocessing-in-python-set-1/)
      [Spacy 101: everything you need to know (20 min)](https://spacy.io/usage/spacy-101)
      **Libraries** - `regex`, `NLTK`, `spaCy`
      **Key patterns:** removing special characters, normalizing whitespace, lowercasing, removing stopwords
      (Extracted text is often messy — extra whitespace, headers/footers, artifacts from conversion)

- [ ] **Splitting large documents**
      [Greg Kamradt: The 5 Levels Of Text Splitting For Retrieval
 (70 min)](https://www.youtube.com/watch?v=8OJC21T2SL4). Watch at least the first 20 minutes for core concepts
      **Key concepts:** why split documents (size limits, focused processing), fixed-size vs semantic splitting, overlap between chunks
      (AI models have input size limits — large documents must be split into smaller pieces for processing)


**Check Yourself:**
- [ ] You receive a 50-page PDF contract and need to extract all the text. What library would you use and what challenges might you encounter?
- [ ] Your extracted text has inconsistent whitespace, random line breaks, and header/footer artifacts on every page. How do you clean this up?
- [ ] A Word document contains both paragraphs and tables with important data. How do you extract both?
- [ ] You need to process a 200-page technical manual, but your processing tool can only handle 5 pages worth of text at a time. What's your approach?
- [ ] Your text file displays "donâ€™t" instead of "don't" when you read it. What's the problem and how do you fix it?
- [ ] What's the difference between NLTK and spaCy? When might you choose one over the other?

### 3.3 SQL Basics (~3 hrs)

> AI pipelines often need data from databases — customer records, transactions, logs. You need enough SQL to extract what you need and get it into Python.

- [ ] **SQL refresher (if needed)**
      [Kevin Stratvert: SQL tutorial for beginners (45 min)](https://www.youtube.com/watch?v=h0nxCDiD-zg)
      [Alternative] [ThoughtSpot SQL Tutorial (1-2 hrs)](https://www.thoughtspot.com/sql-tutorial)
      **Key patterns:** SELECT, WHERE, JOIN (INNER, LEFT), GROUP BY, ORDER BY, LIMIT
      (Skip if you're already comfortable with SQL)

- [ ] **sqlite3 (built-in Python library)**
      [Python docs: sqlite3 - DB.API 2.0 interface for SQLite datagbase (20 min)](https://docs.python.org/3/library/sqlite3.html)
      **Key patterns:** connecting, executing queries, fetching results, parameterized queries
      (Great for prototyping, local development, small datasets — no server needed)

- [ ] **SQLAlchemy (industry standard ORM)**
      [ArjanCodes: SQLAlchemy: The BEST SQL Database Library in Python (17 min)](https://www.youtube.com/watch?v=aAy-B6KPld8)
      **Key patterns:** engine creation, connection, executing raw SQL, basic ORM usage
      (Production standard — works with PostgreSQL, MySQL, SQLite, and more)
      
      Reference: [SQLAlchemy Unified Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)

- [ ]  **PostgreSQL connectivity (reference)**
      [psycopg 3 Documentation: Getting Started (20 min)](https://www.psycopg.org/psycopg3/docs/basic/usage.html)
      **Library:** `psycopg` (latest) or `psycopg2` (legacy, widely used)
      **Key patterns:** connecting, executing queries, transactions, connection pooling
      (PostgreSQL is the most common production database — psycopg is the standard Python driver)
      **Note:** SQLAlchemy uses psycopg under the hood for PostgreSQL connections

**Check Yourself:**
- [ ] You need to get all customers from Germany who made purchases over $100 last month. Write the SQL query.
- [ ] You're prototyping locally and need a simple database without setting up a server. What would you use?
- [ ] How do you execute a SQL query in Python and get results as a list of tuples?
- [ ] Why should you use parameterized queries instead of string formatting when including user input in SQL?
- [ ] You have a complex SQL query that returns 100K rows. What's the fastest way to get this into a Pandas DataFrame?
- [ ] Your production app uses PostgreSQL. What library would you use to connect from Python?

### Level 3 Checkpoint

After completing Level 3, you should be able to:

**Data Quality (3.1)**
- [ ] Identify and handle missing values using appropriate strategies (drop, impute, fill)
- [ ] Detect and remove duplicate records
- [ ] Identify outliers using statistical methods (IQR, Z-score) and decide how to handle them
- [ ] Normalize or standardize data and know when to use which approach
- [ ] Validate incoming data with Pydantic schemas

**Unstructured Data (3.2)**
- [ ] Extract text from PDFs and Word documents using appropriate libraries
- [ ] Clean extracted text (whitespace, special characters, artifacts)
- [ ] Split large documents into smaller chunks for processing
- [ ] Scrape and extract content from web pages
- [ ] Perform basic image preprocessing (resize, convert formats)

**SQL & Databases (3.3)**
- [ ] Write SQL queries to extract specific data (SELECT, JOIN, WHERE, GROUP BY)
- [ ] Connect to databases from Python using sqlite3 or SQLAlchemy
- [ ] Use parameterized queries to prevent SQL injection
- [ ] Load query results into Pandas DataFrames

**Practical Integration**
- [ ] Given a messy PDF document, extract text, clean it, and split it into processable chunks
- [ ] Given a database with customer data, write a query to extract relevant records and load them into Python for analysis
---


## Level 4: Cloud AI Services (~5-6w total)

> **Note:** Use your Azure $200 credits for hands-on practice. 

### 4.1 Cloud AI Services Overview (~6-7 hrs)

> Why cloud AI services? They provide pre-trained models via APIs — no ML expertise needed. You send data, get predictions back. Ideal for common tasks (OCR, sentiment, speech) where building from scratch would take months.

> Each module contains a practical exercise for around 30 min, so the theory is about 20-30 minutes long. Check yourself questions are included in every module in 'Module assessment' section
- [ ] [Get started with AI in Microsoft Foundry (40 min)](https://learn.microsoft.com/en-us/training/modules/get-started-ai-in-foundry/)
- [ ] [Get started with generative AI in Microsoft Foundry (47 min)](https://learn.microsoft.com/en-us/training/modules/get-started-generative-ai-azure/)
- [ ] [Get started with natural language processing in Microsoft Foundry (40 min)](https://learn.microsoft.com/en-us/training/modules/get-started-language-azure/)
- [ ] [Get started with speech in Microsoft Foundry (30 min)](https://learn.microsoft.com/en-us/training/modules/recognize-synthesize-speech/)
- [ ] [Get started with computer vision in Microsoft Foundry (45 min)](https://learn.microsoft.com/en-us/training/modules/get-started-computer-vision-azure/)
- [ ] [Get started with AI-powered information extraction in Microsoft Foundry (1 hr)](https://learn.microsoft.com/en-us/training/modules/ai-information-extraction/)
- [ ] [Optional] [John Savill: Azure AI Foundry overview (87 min)](https://www.youtube.com/watch?v=Sq8Cq7RZM2o)

**Reference:** [Microsoft Learn: Introduction to Azure AI Services (learning path) (10 hr)](https://learn.microsoft.com/en-us/training/paths/get-started-with-artificial-intelligence-on-azure/)

### 4.2 Document & Vision Services (~6-7 hrs)
**Reference:** [Microsoft Learn: Develop computer vision solutions in Azure (6.5 hrs)](https://learn.microsoft.com/en-us/training/paths/create-computer-vision-solutions-azure-ai/):
- [ ] [Analyze images (44 min)](https://learn.microsoft.com/en-us/training/modules/analyze-images/)
- [ ] [Read text in images (42 min)](https://learn.microsoft.com/en-us/training/modules/read-text-images-documents-with-computer-vision-service/)
- [ ] [Detect, analyze, and recognize faces (54 min)](https://learn.microsoft.com/en-us/training/modules/detect-analyze-recognize-faces/)
- [ ] [Classify images (1 hr)](https://learn.microsoft.com/en-us/training/modules/classify-images/)
- [ ] [Object detection in images (1 hr)](https://learn.microsoft.com/en-us/training/modules/detect-objects-images/)
- [ ] [Analyze video (48 min)](https://learn.microsoft.com/en-us/training/modules/analyze-video/)
- [ ] [Develop a vision-enabled generative AI application (43 min)](https://learn.microsoft.com/en-us/training/modules/develop-generative-ai-vision-apps/)
- [ ] [Generate images with AI (33 min)](https://learn.microsoft.com/en-us/training/modules/generate-images-azure-openai/)

### 4.3 Language & Speech Services (~8-9 hrs)
**Reference:** [Microsoft Learn: Develop AI agents on Azure (8 hrs)](https://learn.microsoft.com/en-us/training/paths/develop-ai-agents-on-azure/)

- [ ] [Analyze text with Azure Language (54 min)](https://learn.microsoft.com/en-us/training/modules/analyze-text-ai-language/)
- [ ] [Create question answering solutions with Azure Language (46 min)](https://learn.microsoft.com/en-us/training/modules/create-question-answer-solution-ai-language/)
- [ ] [Build a conversational language understanding model (1 hr)](https://learn.microsoft.com/en-us/training/modules/build-language-understanding-model/)
- [ ] [Create custom text classification solutions (1 hr)](https://learn.microsoft.com/en-us/training/modules/custom-text-classification/)
- [ ] [Custom named entity recognition (53 min)](https://learn.microsoft.com/en-us/training/modules/custom-name-entity-recognition/)
- [ ] [Translate text with Azure Translator service (48 min)](https://learn.microsoft.com/en-us/training/modules/translate-text-with-translator-service/)
- [ ] [Create speech-enabled apps with Microsoft Foundry (53 min)](https://learn.microsoft.com/en-us/training/modules/create-speech-enabled-apps/)
- [ ] [Translate speech with the Azure Speech service (47 min)](https://learn.microsoft.com/en-us/training/modules/translate-speech-speech-service/)
- [ ] [Develop an audio-enabled generative AI application (43 min)](https://learn.microsoft.com/en-us/training/modules/develop-generative-ai-audio-apps/)
- [ ] [Develop an Azure AI Voice Live agent (51 min)](https://learn.microsoft.com/en-us/training/modules/develop-voice-live-agent/)

### 4.4 Develop AI agents on Azure (8 hrs)
**Reference:** [Microsoft Learn: Develop AI agents on Azure (8 hrs)](https://learn.microsoft.com/en-us/training/paths/develop-language-solutions-azure-ai/)

- [ ] [Get started with AI agent development on Azure (49 min)](https://learn.microsoft.com/en-us/training/modules/ai-agent-fundamentals/)
- [ ] [Develop an AI agent with Microsoft Foundry Agent Service (55 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agent-azure/)
- [ ] [Develop AI agents with the Microsoft Foundry extension in Visual Studio Code (48 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agents-vs-code/)
- [ ] [Integrate custom tools into your agent (53 min)](https://learn.microsoft.com/en-us/training/modules/build-agent-with-custom-tools/)
- [ ] [Develop a multi-agent solution with Microsoft Foundry Agent Service (46 min)](https://learn.microsoft.com/en-us/training/modules/develop-multi-agent-azure-ai-foundry/)
- [ ] [Integrate MCP Tools with Azure AI Agents (51 min)](https://learn.microsoft.com/en-us/training/modules/connect-agent-to-mcp-tools/)
- [ ] [Develop an AI agent with Microsoft Agent Framework (55 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agent-with-semantic-kernel/)
- [ ] [Orchestrate a multi-agent solution using the Microsoft Agent Framework (73 min)](https://learn.microsoft.com/en-us/training/modules/orchestrate-semantic-kernel-multi-agent-solution/)
- [ ] [Discover Azure AI Agents with A2A (56 min)](https://learn.microsoft.com/en-us/training/modules/discover-agents-with-a2a/)


### 4.5 [Optional] Certifications 
- [ ] [John Savill: AI-900 - Learning About Generative AI (1 hr)](https://www.youtube.com/watch?v=Ch6KE7KxHGM)
- [ ] [John Savill: AI-900 Study Cram v2 (the non-Generative AI bits) (76 min)](https://www.youtube.com/watch?v=bTkUTkXrqOQ)
- [ ] [John Savill's: AI-102 Study Cram (2 hr)](https://www.youtube.com/watch?v=I7fdWafTcPY)


### Level 4 Checkpoint
### Level 4 Checkpoint

**Cloud AI Understanding**
- [ ] A startup asks whether they should build a custom sentiment analysis model or use a cloud service. They need it working in 3 weeks. What do you recommend and why?
- Your company uses AWS but you've learned Azure. How do you find the equivalent AWS service for Azure Document Intelligence?

**Document & Vision (4.2)**
- [ ] Your client has 10,000 scanned invoices and needs to extract vendor names, dates, and totals. Which Azure service do you use and what's your approach?
- [ ] You need to build a system that detects defective products on a manufacturing line from camera images. Which Azure services could help?
- [ ] A retail company wants to analyze customer demographics from store camera footage. What Azure capabilities would you explore, and what ethical considerations should you raise?

**Language & Speech (4.3)**
- [ ] Your team needs to analyze 50,000 customer support tickets to find common complaints and sentiment trends. How would you approach this with Azure AI services?
- [ ] A call center wants to transcribe customer calls and detect when customers are frustrated. Which Azure services would you combine?
- [ ] You're building a multilingual chatbot that needs to understand questions in 10 languages. What Azure services would you use?

**Practical Application**
- [ ] A law firm wants to extract key clauses, dates, and party names from thousands of PDF contracts. Design a solution using Azure AI services.
- [ ] Your manager asks: "How much will it cost to process 100,000 documents per month with Azure Document Intelligence?" How do you find this answer?

**Service categories (reference):**
| Problem | Azure Service | AWS | GCP |
|---------|---------------|-----|-----|
| OCR / Document extraction | Document Intelligence | Textract | Document AI |
| Image analysis | AI Vision | Rekognition | Vision AI |
| Text analysis (NER, sentiment) | AI Language | Comprehend | Natural Language AI |
| Speech-to-text | Speech Service | Transcribe | Speech-to-Text |
| Text-to-speech | Speech Service | Polly | Text-to-Speech |
| LLM / Chat | Azure OpenAI | Bedrock | Vertex AI |
| Translation | Translator | Translate | Translation AI |

**Alternative clouds (reference):**
- [AWS AI/ML Services Overview](https://aws.amazon.com/machine-learning/)
- [Google Cloud AI Services Overview](https://cloud.google.com/products/ai)

---

## 🎯 Capstone Project A: Smart Receipt Processor (~1w) [TO REVIEW THAT SECTION]

**Complete after Level 4**

- [ ] **Task:** Build a system that takes receipt photos, extracts key info (total, date, vendor), saves as JSON
- [ ] **Tech Stack:** Azure Document Intelligence (or AWS Textract) + Python + Azure Blob Storage
- [ ] **Skills:** Cloud AI APIs, data extraction, error handling, cost awareness
- [ ] **Deliverable:** Working demo + GitHub repo

---

## Level 5: LLMs & Prompt Engineering (~2w-3w total)

### 5.1 LLM Fundamentals (~1-2 hrs)

> Building on Level 1.5, we now go deeper into how LLMs work. This understanding helps you make better decisions about model selection, prompt design, and troubleshooting.

**Transformers — Deep Dive**
- [ ] [Review] [3Blue1Brown: Transformers, the tech behind LLMs  (27 min)](https://www.3blue1brown.com/lessons/gpt)
- [ ] [3Blue1Brown: Attention in transformers, step-by-step (26 min)](https://www.3blue1brown.com/lessons/attention)
- [ ] [3Blue1Brown: How might LLMs store facts (22 min)](https://www.3blue1brown.com/lessons/mlp)
      

**Tokenization in Practice**
- [ ] [OpenAI: What are tokens and how to count them? (5 min)](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them); [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)

**Context Windows & Limitations**
- [ ] [IBM Technology: What is a Context Window? Unlocking LLM Secrets (12 min)](https://www.youtube.com/watch?v=-QVoIxEpFkM)
- [ ] [Matt Pocock: Most devs don’t understand how context windows work (10 min)](https://www.youtube.com/watch?v=-uW5-TaVXu4)
      

**Temperature & Sampling Parameters**
- [ ] [IBM: What is LLM Temperature? (5 min)](https://www.ibm.com/think/topics/llm-temperature)
- [ ] [MarbleScience: Softmax - What is the Temperature of an AI? (9 min)](https://www.youtube.com/watch?v=YjVuJjmgclU)
- [ ] [Google Cloud: Experiment with parameter values (5 min)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values)
- [ ] [Alternative] [Prompting Guide AI: LLM Settings (12 min)](https://www.promptingguide.ai/introduction/settings)
      **Key parameters:**
      - Temperature (0 = deterministic, 1+ = creative)
      - Top-p (nucleus sampling)

**Useful links**
- **Frontier models:** [ChatGPT (OpenAI)](https://chatgpt.com/), [Claude (Anthropic)](https://claude.ai/), [Gemini (Google)](https://gemini.google.com/)

- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/leaderboards/models) - the landscape changes fast - know where to check current benchmarks.

**Check Yourself:**
- You're building a customer service bot that needs consistent, predictable answers. What temperature setting would you use and why?
- The same prompt costs $0.01 with GPT-4o but $0.001 with GPT-4o-mini. How do you decide which to use?
- You notice your LLM outputs are repetitive. Which sampling parameter might help?
- "Tokenization" has 3 tokens in GPT-4 but might have different count in Llama. Why does this matter?
- A document has critical information in the middle of 100 pages. Why might the LLM miss it, and how would you address this?

### 5.2 Prompt Engineering & LLM Patterns (~6-7 hrs)

Prompt engineering is how you communicate with LLMs effectively. These techniques range from simple (zero-shot) to advanced (tool use), and you'll often combine them.

**Comprehensive Resources**
- [Learn Prompting](https://learnprompting.org/docs/introduction)
- [Prompt Engineering Guide (community)](https://www.promptingguide.ai/)
- [Anthropic: Prompt Engineering Documentation (30 min)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [OpenAI: Prompt Engineering Guide (20 min)](https://platform.openai.com/docs/guides/prompt-engineering)


**Prompting Techniques basics (~1 hr)**
- [ ] [Learn Prompting: Instructions in prompts (6 min)](https://learnprompting.org/docs/basics/instructions)
- [ ] [Learn Prompting: Roles in Prompts (9 min)](https://learnprompting.org/docs/basics/roles); [Revisitng roles (3 min)](https://learnprompting.org/docs/intermediate/revisiting_roles)
- [ ] [Learn Prompting: From Zero-Shot to Few-Shot (13 min)](https://learnprompting.org/docs/basics/few_shot)
- [ ] [Learn Prompting: Combining Prompting Techniques (4 min)](https://learnprompting.org/docs/basics/combining_techniques)
- [ ] [Learn Prompting: Tips for Writing Better Prompts (5 min)](https://learnprompting.org/docs/basics/ai_prompt_tips)
- [ ] [Learn Prompting: Prompt Priming: Setting Context for AI (11 min)](https://learnprompting.org/docs/basics/priming_prompt)
- [ ] [Learn Prompting: Differences Between Chatbots and LLMs (6 min)](https://learnprompting.org/docs/basics/chatbot_basics)
- [ ] [Learn Prompting: LLM Limitations: When Models and Chatbots Make Mistakes (5 min)](https://learnprompting.org/docs/basics/pitfalls)
- [ ] [Learn Prompting: What Can Generative AI Create Beyond Text? (7 min)](https://learnprompting.org/docs/basics/generative_ai_applications)
- [ ] [Learn Prompting: How to Solve Problems Using Generative AI: A Simple Method (5 min)](https://learnprompting.org/docs/basics/learn_prompting_method)
      
**Advanced Prompting Techniques (40-50 min)**
- [ ] [Learn Prompting: Chain-of-Thought Prompting (4 min)](https://learnprompting.org/docs/intermediate/chain_of_thought)
- [ ] [Learn Prompting: Zero-Shot Chain-of-Thought (3 min)](https://learnprompting.org/docs/intermediate/zero_shot_cot)
- [ ] [Learn Prompting: Self-Consistency (3 min)](https://learnprompting.org/docs/intermediate/self_consistency)
- [ ] [Learn Prompting: Generated Knowledge (5 min)](https://learnprompting.org/docs/intermediate/generated_knowledge)
- [ ] [Learn Prompting: Least-to-Most Prompting (5 min)](https://learnprompting.org/docs/intermediate/least_to_most)
- [ ] [Learn Prompting: Dealing With Long Form Content (3 min)](https://learnprompting.org/docs/intermediate/long_form_content)
- [ ] [Learn Prompting: Aligned Chain-of-Thought (AlignedCoT) (4 min)](https://learnprompting.org/docs/new_techniques/aligned_cot)
- [ ] [Learn Prompting: (Self-Harmonized Chain-of-Thought (ECHO)) (3 min)](https://learnprompting.org/docs/new_techniques/self_harmonized_chain_of_thought)
- [ ] [Learn Prompting: Logic-of-Thought (LoT) (5 min)](https://learnprompting.org/docs/new_techniques/logic_of_thought)
- [ ] [Learn Prompting: Narrative-of-Thought (NoT) (5 min)](https://learnprompting.org/docs/new_techniques/narrative_of_thought)
- [ ] [Learn Prompting: Code Prompting (3 min)](https://learnprompting.org/docs/new_techniques/code_prompting)
- [ ] [Prompting Guide AI: ReAct Prompting (10 min)](https://www.promptingguide.ai/techniques/react)

**Alternative** - [PromptingGuide.AI - techniques](https://www.promptingguide.ai/techniques)

**Controlling Output Format - JSON mode (20 min)** - force valid JSON output - essential for parsing LLM responses.programmatically
- [ ] [OpenAI: Structured model outputs (10 min)](https://platform.openai.com/docs/guides/structured-outputs)
- [ ]  [Anthropic: Structured outputs (10 min)](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
      

**Function calling / Tool use (30 min)** - LLM decides which function to call and with what arguments - bridges LLMs to external systems
- [ ] [OpenAI: Function Calling (10 min)](https://platform.openai.com/docs/guides/function-calling)
- [ ] [Anthropic: Tool Use (5 min)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [ ] [Gemini: Function calling (15 min)](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)
      

**Check Yourself:**
- [ ] Your LLM keeps returning inconsistent date formats (sometimes "Jan 5", sometimes "2024-01-05"). How do you fix this?
- [ ] You need to extract product name, price, and category from unstructured product descriptions. What combination of techniques would you use?
- [ ] A complex math problem gets wrong answers with zero-shot prompting. What technique should you try first?
- [ ] Your prompt works 80% of the time but fails 20%. How do you improve reliability without changing the core prompt?
- [ ] You're building a chatbot that needs to check inventory, process payments, and send emails. How do you give the LLM these capabilities?
- [ ] When would you use prompt chaining instead of one large prompt?
- You want the LLM to always respond with a specific JSON structure containing "sentiment", "confidence", and "keywords". How do you ensure this?

### 5.3 Local LLMs — Overview (~1 hr)

> This section is reference material. The local LLM landscape evolves rapidly — focus on understanding when and why to use local models, not memorizing specific tools.

**When to use local vs cloud**
| Use Local | Use Cloud |
|-----------|-----------|
| Sensitive data / privacy requirements | Need best-in-class quality |
| High volume (cost optimization) | Quick prototyping |
| Offline / air-gapped environments | No infrastructure to manage |
| Full control over model | Always up-to-date models |

**Running LLMs Locally**
- [ ] [Ollama](https://ollama.ai/) — simplest way to run local LLMs
      [NetworkChuck: host all your AI locally (24 min)](https://www.youtube.com/watch?v=Wjrdr0NU4Sk)

- [ ] [LM Studio](https://lmstudio.ai/) — desktop app with GUI
      User-friendly interface, good for experimentation without CLI

- **Open-source:** [Llama 3 (Meta)](https://github.com/meta-llama/llama3), [Qwen (Alibaba)]
**Popular Open-Source Models (as of 2025)**
- [Llama 3 / 3.1 (Meta)](https://github.com/meta-llama/llama3) — strong general-purpose, various sizes
- [Mistral / Mixtral](https://huggingface.co/mistralai) — efficient, good quality-to-size ratio
- [Qwen 2.5 (Alibaba)](https://github.com/QwenLM/Qwen) — strong multilingual and coding
- [Phi-3 (Microsoft)](https://huggingface.co/collections/microsoft/phi-3) — small but capable

Check current rankings: [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

**Unified APIs (reference)**
- [ ] [OpenRouter](https://openrouter.ai/) — single API for 100+ models (OpenAI, Anthropic, open-source). Useful for comparing models, fallbacks, cost optimization
      
- [ ] [LiteLLM](https://github.com/BerriAI/litellm) — open-source LLM gateway. Useful for self-hosted proxy, load balancing, spend tracking, OpenAI-compatible format

**Check Yourself:**
- [ ] Your client has strict data privacy requirements and cannot send data to external APIs. What are your options?
- [ ] You want to test your prompt across GPT-4, Claude, and Llama without rewriting code for each API. What tools could help?
- [ ] When would running your own LLM be more cost-effective than using OpenAI's API?
- [ ] You need to deploy an LLM for a high-traffic production application. Which tool is designed for this?

### Level 5 Checkpoint
- [ ] Can design effective prompts and get structured JSON outputs
- [ ] Know when to use different prompting techniques

---

## Level 6: RAG & Vector Databases (~3w total)

> **Why this matters:** RAG is the #1 pattern for enterprise AI. This is where most business value is created.

### 6.1 Embeddings (~3 hrs)

> Embeddings convert text (or images, audio) into numerical vectors that capture meaning. Similar concepts end up close together in vector space. This is the foundation of semantic search and RAG.

**Understanding Embeddings**
- [ ] [Review] [3Blue1Brown: Word embedding — from GPT video (from 12:27) (15 min)](https://youtu.be/wjZofJX0v4M?si=i-7iJu873HEZoFVH&t=747)
- [ ] [Computerphile: "Word Embeddings" (17 min)](https://www.youtube.com/watch?v=gQddtTdmG_8)
- [ ] [StatQuest: Word Embedding and Word2Vec, Clearly Explained (16 min)](https://www.youtube.com/watch?v=viZrOnJclY0)
- [ ] [IBM Technology: What are Word Embeddings? (8 min)](https://www.youtube.com/watch?v=wgfSDrqYMJ4)
- [ ] [Optional] [OpenAI: Embeddings Guide (15 min)](https://platform.openai.com/docs/guides/embeddings)

For reference:
- [Sentence Transformers (open source)](https://www.sbert.net/docs/quickstart.html) ([Github](https://github.com/huggingface/sentence-transformers))
      Run locally, no API costs — great for development and privacy-sensitive use cases

**Embedding Models Comparison**
| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| text-embedding-3-small (OpenAI) | 1536 | 8K | Cost-effective, general use |
| text-embedding-3-large (OpenAI) | 3072 | 8K | Higher quality, more expensive |
| Cohere embed-v3 | 1024 | 512 | Multilingual, search-optimized |
| BGE (open source) | 768-1024 | 512 | Self-hosted, no API costs |
| all-MiniLM-L6-v2 (sentence-transformers) | 384 | 256 | Fast, lightweight, local |

Check current rankings: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

**Semantic Similarity** - how to measure similarity between embeddings
- [ ] [StatQuest: Cosine Similarity (10 min)](https://www.youtube.com/watch?v=e9U0QAFbfLI)
      **Key concepts:**
      - Cosine similarity: measures angle between vectors (-1 to 1)
      - Euclidean distance: measures straight-line distance
      - Dot product: fast similarity for normalized vectors
      - Most vector DBs use cosine similarity by default

**Check Yourself:**
- [ ] In simple terms, what is an embedding and why is it useful for search?
- [ ] You have two sentences: "The cat sat on the mat" and "A feline rested on the rug." Would their embeddings be close or far apart? Why?
- [ ] Your embedding model outputs 1536-dimensional vectors. What does each dimension represent?
- [ ] You need to embed 1 million documents. Would you use OpenAI's API or sentence-transformers? What factors influence this decision?
- [ ] What's the difference between cosine similarity and Euclidean distance? When might you prefer one over the other?
- [ ] How do you check if your embedding model is good for your specific use case?


### 6.2 Vector Databases (~2 hrs)

> Vector databases are specialized for storing and searching embeddings. Regular databases can't efficiently search "find me similar vectors" across millions of items — vector DBs are built exactly for this.

**Understanding Vector Databases**
- [ ] [Fireship: Vector databases are so hot right now. WTF are they? (3 min)](https://www.youtube.com/watch?v=klTvEwg3oJ4)
- [ ] [IBM Technology: What is a Vector Database? Powering Semantic Search & AI Applications (8 min)](https://www.youtube.com/watch?v=gl1r1XV0SLw)
- [ ] [Pinecone: What is a Vector Database? (20 min)](https://www.pinecone.io/learn/vector-database/)
- [ ] [Alternative] [IBM Technology: What is a Vector Database? (8 min)](https://www.youtube.com/watch?v=t9IDoenf-lo)
- [ ] [Alternative] [The ML TechLead: Understanding How Vector Databases Work! (13 min)](https://www.youtube.com/watch?v=035I2WKj5F0)

**Core Operations**
All vector databases share these fundamental operations:
- **Insert/Upsert** — add vectors with metadata
- **Query** — find top-K similar vectors
- **Filter** — combine similarity search with metadata filters
- **Delete** — remove vectors by ID or filter

**Vector Database Options**

| Database | Type | Best For | Notes |
|----------|------|----------|-------|
| [Chroma](https://www.trychroma.com/home?) | Open source | Development, prototyping | Simple API, runs in-memory or persistent |
| [Pinecone](https://www.pinecone.io/) | Managed cloud | Production, zero-ops | Free tier available, easy scaling |
| [Qdrant](https://qdrant.tech/) | Open source / Cloud | Production, self-hosted | Rust-based, high performance |
| [Weaviate](https://weaviate.io/) | Open source / Cloud | Hybrid search, multimodal | Built-in vectorization |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension | Existing Postgres infrastructure | No new database to manage |
| [Milvus](https://milvus.io/) | Open source | Large scale, enterprise | Distributed, handles billions of vectors |

**References:**
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Weaviate Documentation](https://docs.weaviate.io/weaviate)
- [Milvus Documentation](https://milvus.io/docs)

**Check Yourself:**
- [ ] Why can't you just store embeddings in PostgreSQL without pgvector and search efficiently?
- [ ] Your production system needs 99.9% uptime and you don't want to manage infrastructure. What's your best option?
- [ ] You have an existing PostgreSQL database with user data. You want to add semantic search without introducing a new database. What do you use?
- [ ] What's the difference between a similarity search and a filtered similarity search? Give an example where filtering matters.
- [ ] Your vector database returns the 10 most similar documents, but some aren't very relevant. What could you do?

### 6.3 RAG Architecture (~4-5 hrs)

> RAG (Retrieval-Augmented Generation) combines search with LLMs. Instead of relying solely on the LLM's training data, you retrieve relevant documents and include them in the prompt. This grounds the LLM's response in your actual data.

**Why RAG?**
| Approach | Pros | Cons |
|----------|------|------|
| Fine-tuning | Baked-in knowledge, fast inference | Expensive, static, hallucination risk |
| Long context | Simple, no infrastructure | Costly per request, attention limits |
| RAG | Fresh data, source attribution, cost-effective | More components, retrieval quality matters |

RAG wins for most enterprise use cases: knowledge bases, document Q&A, support systems.

**Understanding RAG**
- [ ] [Computerphile: A Helping Hand for LLMs (Retrieval Augmented Generation) (14 min)](https://www.youtube.com/watch?v=of4UDMvi2Kw)
- [ ] [KodeKloud: RAG Explained for beginners (10 min)](https://www.youtube.com/watch?v=_HQ2H_0Ayy0)
- [ ] [IBM Technology: What is RAG? (8 min)](https://www.youtube.com/watch?v=T-D1OfcDW1M)
- [ ] [Anthropic: What is RAG? (15 min)](https://www.anthropic.com/news/contextual-retrieval)

Optional deep dive (at least one is recommended to watch):
- [ ] [freeCodeCamp.org: Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer (2.5 hrs)](https://www.youtube.com/watch?v=sVcwVQRHIc8)
- [ ] [freeCodeCamp.org: Vector Search RAG Tutorial – Combine Your Data with LLMs with Advanced Search (70 min)](https://www.youtube.com/watch?v=JEBDfGqrAUA)
- [ ] [Underfitted: Building a RAG application from scratch using Python, LangChain, and the OpenAI API (72 min)](https://www.youtube.com/watch?v=BrsocJb-fAo)


- **Ingestion (indexing):** Load documents → chunk → embed → store in vector DB
- **Retrieval (query time):** Embed query → search vector DB → get top-K chunks
- **Generation:** Combine retrieved chunks + query in prompt → LLM generates answer

**Chunking strategies** (revisit from Level 3.2)
- Chunk size: 256-1024 tokens typical
- Overlap: 10-20% prevents cutting important context
- Metadata: preserve source, page number, section for filtering

**Check Yourself:**
- [ ] Explain the RAG pipeline in your own words: what happens from user question to final answer?
- [ ] Your company has 10,000 internal documents that update weekly. Why is RAG better than fine-tuning here?
- [ ] A user asks a question and gets a wrong answer. How do you debug whether the problem is retrieval or generation?
- [ ] You retrieve 5 chunks but only 2 are relevant. What could be causing this and how would you improve it?
- [ ] Why must you use the same embedding model for documents and queries?
- [ ] Your chunks are 2000 tokens each but your retrieved context seems to miss important details. What might you change?

### 6.4 RAG Optimization & Evaluation (~3-4 hrs)

> Basic RAG gets you 70% there. These techniques and evaluation methods get you to production quality. Focus on hybrid search and re-ranking first — they have the highest impact.

**Why Basic RAG Isn't Enough**
- Vector search misses exact keyword matches ("Policy ABC-123")
- Top-K retrieval returns similar but not always relevant chunks
- Without evaluation, you're guessing if changes help

**Hybrid Search (Keyword + Semantic)**
- [ ] [Pinecone: Hybrid Search Explained (20 min)](https://www.pinecone.io/learn/hybrid-search-intro/)
- [ ] [Qdrant: Hybrid Search with Sparse and Dense Vectors (20 min)](https://qdrant.tech/articles/hybrid-search/)
      
**Key concept:** Combine BM25/keyword search with vector search
- Vector search: "similar meaning"
- Keyword search: "exact matches"
- Together: best of both worlds

**Re-ranking**
- [ ] [Pinecone: Rerankers for Better RAG (20 min)](https://www.pinecone.io/learn/series/rag/rerankers/)
      
**Key concept:** Retrieve more candidates (top-20), then re-rank with a cross-encoder to get best top-5
- First-stage: fast, cheap retrieval (bi-encoder)
- Second-stage: accurate re-ranking (cross-encoder)

**Other Techniques (reference)**
- **Query transformation:** Rewrite query for better retrieval (HyDE, query expansion)
- **Parent-child chunking:** Store small chunks, retrieve parent context
- **Multi-query retrieval:** Generate multiple query variants, combine results

[Learn more: LlamaIndex Advanced RAG Techniques (10 min)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

**RAG Evaluation**

> "If you can't measure it, you can't improve it." RAG evaluation is uniquely challenging - you need to evaluate both retrieval AND generation.

**What to Measure**
| Metric | Measures | Question It Answers |
|--------|----------|---------------------|
| Context Relevance | Retrieval | Did we find the right documents? |
| Faithfulness | Generation | Is the answer grounded in retrieved context? |
| Answer Relevance | End-to-end | Does the answer address the question? |

**RAGAS Framework** - Industry standard for RAG evaluation — provides faithfulness, relevance, context metrics
- [ ] [RAGAS Documentation: Metrics (20 min)](https://docs.ragas.io/en/latest/concepts/metrics/overview/)
- [ ] [RAGAS: Getting Started (15 min)](https://docs.ragas.io/en/latest/getstarted/)

**LLM-as-Judge**
- [ ] [Anthropic: Using Claude for Evaluation (15 min)](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests)
- [ ] [OpenAI Cookbook: LLM-as-Judge (20 min)](https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization)
      
**Key concept:** Use an LLM to evaluate LLM outputs
- Scalable (no human labeling needed)
- Consistent (same criteria every time)
- Practical for iteration
      
**Evaluation Workflow**
```
1. Create test dataset (50-100 question-answer pairs)
2. Run RAG pipeline on test questions
3. Evaluate with RAGAS / LLM-as-judge
4. Identify failure patterns
5. Improve (chunking, retrieval, prompts)
6. Re-evaluate → iterate
```

**Reference:**
- [DeepEval Documentation](https://deepeval.com/docs/getting-started) — alternative evaluation framework

**Check Yourself:**
- [ ] Your RAG system can't find documents when users search by product ID like "SKU-12345". What's likely wrong and how do you fix it?
- [ ] You retrieve 10 chunks but the most relevant one is ranked #8. What technique would help?
- [ ] How would you measure if your RAG answers are actually grounded in the retrieved documents?
- [ ] You want to evaluate your RAG system but don't have time to manually review 1000 responses. What approach would you use?
- [ ] Your RAGAS faithfulness score is 0.6. What does this mean and what might you investigate?
- [ ] You improved your chunking strategy. How do you know if it actually helped?

### Level 6 Checkpoint

**Embeddings (6.1)**
- [ ] In simple terms, what is an embedding and why does it enable semantic search?
- [ ] You need to embed 1 million documents for a startup with limited budget. Would you use OpenAI's API or an open-source model like sentence-transformers? What factors drive this decision?
- [ ] Two sentences have cosine similarity of 0.92. What does this tell you?

**Vector Databases (6.2)**
- [ ] Why can't a regular PostgreSQL database efficiently handle "find similar vectors" queries across millions of embeddings?
- [ ] You're prototyping a RAG system this week. Which vector database would you start with and why?
- [ ] Your company already runs PostgreSQL in production. How would you add vector search with minimal infrastructure changes?

**RAG Architecture (6.3)**
- [ ] Walk through the RAG pipeline: what happens from a user's question to the final answer?
- [ ] Your company has 10,000 documents that update weekly. Why is RAG better than fine-tuning for this use case?
- [ ] A user gets a wrong answer. How do you determine whether the problem is retrieval (wrong documents) or generation (wrong interpretation)?
- [ ] Why must you use the same embedding model for both documents and queries?

**RAG Optimization & Evaluation (6.4)**
- [ ] Users search for "Policy ABC-123" but RAG returns nothing relevant. What's likely wrong and how do you fix it?
- [ ] You retrieve 10 chunks but the best one is ranked #8. What technique would improve this?
- [ ] You want to evaluate 1000 RAG responses but can't review them manually. What approach would you use?
- [ ] Your RAGAS faithfulness score is 0.6. What does this indicate and what would you investigate?
- [ ] You changed your chunking strategy from 512 to 256 tokens. How do you measure if this actually improved results?

**Practical Integration**
- [ ] Design a RAG system for a legal firm that needs to answer questions about 50,000 contracts. What embedding model, vector database, and retrieval strategy would you recommend?
- [ ] Your RAG system is live but users complain answers are sometimes wrong. Outline your debugging and evaluation approach.

---

## Level 7: Building AI Applications (~3w-4w total)

> **Framework Note:** The OpenAI Python SDK is the go-to library for most LLM work. You may prototype with LangChain-like frameworks to learn concepts quickly, but be aware these are often over-abstracted — you'll spend time fighting the framework rather than building. Most practitioners shift to lower-level SDKs (OpenAI, Anthropic, LiteLLM) or direct API calls once they move beyond prototyping. Each major provider has its own native API with unique features not fully accessible through compatibility layers.

### 7.1 Orchestration & Agents (~7 hrs)

> Orchestration frameworks help you build complex AI workflows — chains of prompts, tool usage, and decision-making. Agents are AI systems that can reason about which actions to take and execute them autonomously.

**Understanding Agents**
- [ ] [IBM Technology: What are AI Agents? (12 min)](https://www.youtube.com/watch?v=F8NKVhkZZWI)
- [ ] [Anthropic: Tips for building AI agents (18 min)](https://www.youtube.com/watch?v=LP5OCa20Zpg)
- [ ] [Anthropic: What are AI Agents? (15 min)](https://www.anthropic.com/research/building-effective-agents)
- [ ] [Review from Level 1] [Jeff Su: AI Agents, Clearly Explained (10 min)](https://www.youtube.com/watch?v=FwOTs4UxQS4)

**LangChain Basics**
- [ ] [IBM Technology: What is LangChain? (8 min)](https://www.youtube.com/watch?v=1bUy-1hGZpI)
- [ ] [Rabbitmetrics: LangChain Explained | QuickStart Tutorial for Beginners (13 min)](https://www.youtube.com/watch?v=aywZrzNaKjs)
- [ ] [LangChain: Introduction (3 min)](https://python.langchain.com/docs/get_started/introduction) + [QuickStart (10 min)](https://docs.langchain.com/oss/python/langchain/quickstart)
- [ ] [freeCodeCamp: LangChain Tutorial for Beginners (90 min)](https://www.youtube.com/watch?v=HSZ_uaif57o)

- [ ] [LangChain: RAG Tutorial (30 min)](https://python.langchain.com/docs/tutorials/rag/)
- [ ] [pixegami: RAG + Langchain Python Project: Easy AI/Chat For Your Docs (16 min)](https://www.youtube.com/watch?v=tcqEUSNCn8I)


**LangGraph for Workflows**
- [ ] [Tech with Time: LangGraph Tutorial - How to Build Advanced AI Agent Systems (47 min)](https://www.youtube.com/watch?v=1w5cCXlh7JQ)
- [ ] [IBM Technology: LangChain vs LangGraph: A Tale of Two Frameworks (10 min)](https://www.youtube.com/watch?v=qAF1NjEVHhY)
- [ ] [LangChain: LangGraph Quickstart (5 min)](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [ ] [Sam Witteveen: LangGraph Explained (25 min)](https://www.youtube.com/watch?v=R8KB-Zcynxc)

**Building Agents**
- [ ] [Tech with Tim: Build an AI Agent From Scratch in Python - Tutorial for Beginners (34 min)](https://www.youtube.com/watch?v=bTMPwUgLZf0)

**MCP (Model Context Protocol)**
- [ ] [IBM Technology: What is MCP? Integrate AI Agents with Databases & APIs (4 min)](https://www.youtube.com/watch?v=eur8dUO9mvE)
- [ ] [Fireship: Claude's Model Context Protocol is here... Let's test it (8 min)](https://www.youtube.com/watch?v=HyzlYwjoXOQ)
- [ ] [Greg Isenber: Model Context Protocol (MCP), clearly explained (why it matters) (20 min)](https://www.youtube.com/watch?v=7j_NE6Pjv-E)


**Reference:**
- [Anthropic: MCP Documentation](https://modelcontextprotocol.io/)
- [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/understanding/agent/)
- [AutoGen (Microsoft)](https://microsoft.github.io/autogen/)

**Check Yourself:**
-[ ] What's the difference between a chain and an agent? When would you use each?
- [ ] Your AI workflow needs to: search documents → analyze results → decide if more search needed → generate report. Is this a chain or agent pattern? Why?
- [ ] An agent keeps calling the same tool in a loop and never finishes. What might be wrong?
- [ ] You're building integrations for 5 different LLM providers. How could MCP help?
- [ ] When would you choose LangGraph over basic LangChain chains?
- [ ] Your agent needs to book a flight: search flights → select one → enter passenger details → confirm payment. How would you structure this with human-in-the-loop approval?

### 7.2 AI Agents on Azure (~8 hrs)

> After learning agent concepts with LangChain/LangGraph (7.1), this section shows how to build and deploy agents using Azure's managed services. Azure AI Foundry provides production-ready infrastructure for agents without managing your own orchestration layer.

**When to use Azure AI Agents vs LangChain:**
| Use Azure AI Agents | Use LangChain/LangGraph |
|---------------------|-------------------------|
| Already on Azure infrastructure | Cloud-agnostic needs |
| Want managed scaling & monitoring | Need full customization |
| Enterprise compliance requirements | Rapid prototyping |
| Production deployment with minimal ops | Complex custom workflows |

**Reference:** [Microsoft Learn: Develop AI agents on Azure (8 hrs)](https://learn.microsoft.com/en-us/training/paths/develop-ai-agents-on-azure/)

**Develop AI agents on Azure**
- [ ] [Get started with AI agent development on Azure (49 min)](https://learn.microsoft.com/en-us/training/modules/ai-agent-fundamentals/)
- [ ] [Develop an AI agent with Microsoft Foundry Agent Service (55 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agent-azure/)
- [ ] [Develop AI agents with the Microsoft Foundry extension in VS Code (48 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agents-vs-code/)
- [ ] [Integrate custom tools into your agent (53 min)](https://learn.microsoft.com/en-us/training/modules/build-agent-with-custom-tools/)
- [ ] [Develop a multi-agent solution with Microsoft Foundry Agent Service (46 min)](https://learn.microsoft.com/en-us/training/modules/develop-multi-agent-azure-ai-foundry/)
- [ ] [Integrate MCP Tools with Azure AI Agents (51 min)](https://learn.microsoft.com/en-us/training/modules/connect-agent-to-mcp-tools/)
- [ ] [Develop an AI agent with Microsoft Agent Framework (55 min)](https://learn.microsoft.com/en-us/training/modules/develop-ai-agent-with-semantic-kernel/)
- [ ] [Orchestrate a multi-agent solution using the Microsoft Agent Framework (73 min)](https://learn.microsoft.com/en-us/training/modules/orchestrate-semantic-kernel-multi-agent-solution/)
- [ ] [Discover Azure AI Agents with A2A (Agent-to-Agent) (56 min)](https://learn.microsoft.com/en-us/training/modules/discover-agents-with-a2a/)

**Check Yourself:**
- [ ]  You built a LangGraph agent locally. What benefits would you get from migrating it to Azure AI Foundry?
- [ ] Your agent needs to query a company database and call an external API. How do you integrate these as custom tools in Azure?
- [ ] You're building a customer support system where one agent handles refunds and another handles technical issues. How would you architect this as a multi-agent solution?
- [ ] What is A2A (Agent-to-Agent) and when would you use it?
- [ ] Your company already uses MCP servers for tool integration. How does Azure AI Agents support this?

### 7.1 API Development (~5.5 hrs)
- [ ] FastAPI fundamentals
      FastAPI Documentation
      https://fastapi.tiangolo.com/tutorial/

- [ ] Async/await in Python
      Real Python: Async IO
      https://realpython.com/async-io-python/
      (Essential for high-concurrency AI apps)

- [ ] Pydantic for request/response
      Request validation, response models, error handling

- [ ] Streaming responses
      Server-Sent Events for LLM streaming output

### 7.2 Orchestration & Agents (~7 hrs)

**LangChain RAG**
- [ ] LangChain basics
      LangChain Documentation
      https://python.langchain.com/docs/get_started/introduction

- [ ] LangGraph for workflows
      LangGraph Documentation
      https://langchain-ai.github.io/langgraph/
      (Stateful, multi-step AI workflows)

- [ ] Agents and tool use
      Building agents that can use multiple tools

- [ ] (Optional) Deep Agents — LangChain research project
      https://github.com/langchain-ai/deepagents
      Explore this if you want to understand how agents like Claude Code, Cursor, Cline work under the hood

- [ ] MCP (Model Context Protocol)
      Anthropic MCP Documentation
      https://modelcontextprotocol.io/
      (Standardized tool integration)

### 7.3 Containerization (~4 hrs)
- [ ] Docker basics
      Docker: Getting Started
      https://docs.docker.com/get-started/

- [ ] Dockerfile for AI apps
      Python base images, dependencies, environment variables

- [ ] Docker Compose
      Multi-container setups (app + vector DB + redis)
      https://docs.docker.com/compose/

### 7.4 Prototyping (~1.5 hrs)
- [ ] Streamlit for quick demos
      Streamlit Documentation
      https://docs.streamlit.io/

- [ ] Gradio alternative
      Gradio Documentation
      https://www.gradio.app/docs

### 7.5 Rapid Prototyping — No-Code/Low-Code (~1.5 hrs)
> For quick demos to stakeholders or hypothesis validation in 1-2 days

- [ ] OpenAI Agent Builder (visual agent builder)
      https://platform.openai.com/agent-builder
      Build and test agents without writing code

- [ ] n8n (workflow automation with AI nodes)
      https://n8n.io/
      Self-hostable, connect AI to any API/service visually

### 7.6 Security Basics (~3 hrs)
- [ ] Authentication for AI APIs
      OAuth2, API keys, JWT tokens in FastAPI

- [ ] Input validation and sanitization
      Preventing prompt injection via input validation

- [ ] Secure secrets management
      Environment variables, Azure Key Vault, AWS Secrets Manager

### Level 7 Checkpoint
- [ ] Can build containerized FastAPI app serving AI model
- [ ] Can handle authentication and stream responses

---

## 🎯 Capstone Project B: HR Policy Chatbot (~3w) [TO REVIEW THAT SECTIOn]

**Complete after Level 7**

- [ ] **Task:** Build RAG chatbot answering questions about company HR policies from PDF handbook
- [ ] **Tech Stack:** PDF parsing + Chunking + Vector DB (Chroma/Pinecone) + FastAPI + Docker + Streamlit UI
- [ ] **Skills:** Full RAG pipeline, API development, containerization, async, evaluation
- [ ] **Deliverable:** Deployed application + GitHub repo + demo video

---

## Level 8: Production & Operations (~3w total)

### 8.1 Deployment Strategies (~4 hrs)
- [ ] Batch vs Real-time inference
      When to use each: scheduled jobs vs API endpoints

- [ ] Deployment options
      Azure App Service, AWS Lambda, Cloud Run, self-hosted

- [ ] CI/CD for AI applications
      GitHub Actions, Azure DevOps
      Model versioning, automated testing

### 8.2 Monitoring & Observability (~4 hrs)
- [ ] Logging for AI systems
      Structured logging, request/response logging, token usage

- [ ] Tracing tools
      LangFuse (open-source, self-hostable): https://langfuse.com/
      Arize Phoenix: https://phoenix.arize.com/
      (Essential for debugging chains/agents)
      LangFuse also covers testing, prompt versioning, and evaluation — all-in-one observability

- [ ] Metrics to track
      Latency, error rate, token usage, cost per request

- [ ] Data drift detection
      Monitoring input distribution changes over time

### 8.3 User Feedback Loops (~3 hrs)
- [ ] Capturing user feedback
      Thumbs up/down, ratings, corrections in UI

- [ ] Using feedback for improvement
      Prompt refinement, fine-tuning data, evaluation datasets

- [ ] Analytics and reporting
      Track success metrics, identify failure patterns

### 8.4 Optimization (~3 hrs)
- [ ] Caching strategies
      Prompt caching, response caching, semantic caching
      Redis for caching layer

- [ ] Cost optimization
      Model selection, prompt compression, batching requests

- [ ] A/B testing for AI
      Testing different prompts, models, RAG configurations

### 8.5 Debugging AI Systems (~3 hrs)
- [ ] Why RAG returns bad results
      Systematic debugging: chunking → embeddings → retrieval → generation

- [ ] LLM output debugging
      Temperature issues, prompt problems, context overflow

- [ ] Using traces effectively
      LangFuse traces for step-by-step debugging

### Level 8 Checkpoint
- [ ] Can deploy AI app with proper monitoring
- [ ] Can implement caching and user feedback collection
- [ ] Can systematically debug AI issues

---

## Level 9: AI Safety & Ethics (~1w total)

### 9.1 AI Risks (~3 hrs)
- [ ] Hallucinations
      Detection techniques, mitigation, grounding with RAG
      Anthropic: Reducing Hallucinations
      https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering

- [ ] Bias in AI systems
      Sources of bias, testing, mitigation
      Google: Responsible AI Practices
      https://ai.google/responsibility/responsible-ai-practices/

- [ ] Prompt injection attacks
      OWASP: LLM Top 10
      https://owasp.org/www-project-top-10-for-large-language-model-applications/

### 9.2 Responsible AI (~2 hrs)
- [ ] When NOT to use AI
      Google: Rules of ML
      https://developers.google.com/machine-learning/guides/rules-of-ml
      (Sometimes simple rules work better)

- [ ] GDPR and data privacy basics
      Personal data in training, right to explanation, data retention

- [ ] Transparency and explainability
      When users need to understand AI decisions

### 9.3 Business Considerations (~1.5 hrs)
- [ ] Cost estimation for production
      Projecting costs at scale, budget planning

- [ ] Setting realistic expectations
      AI is not magic — communicating limitations to stakeholders

- [ ] Build vs Buy decisions
      When to use off-the-shelf vs custom solutions

### Level 9 Checkpoint
- [ ] Can identify risks in AI system and propose mitigations
- [ ] Can advise when AI is NOT the right solution

---

## Level 10: Fine-tuning LLMs — Optional (~1w-2w total)

> **Note:** This is optional. Most AI Engineer tasks can be solved with prompt engineering and RAG.

### 10.1 When to Fine-tune (~1.5 hrs)
- [ ] Decision framework
      Fine-tuning vs RAG vs Prompt Engineering:
      - Prompt Engineering: first try, fast iteration
      - RAG: need external knowledge, changing data
      - Fine-tuning: specific style, format, or domain expertise

- [ ] Cost-benefit analysis
      Fine-tuning cost vs inference savings, maintenance overhead

### 10.2 Fine-tuning Approaches (~4 hrs)
- [ ] OpenAI fine-tuning API
      OpenAI: Fine-tuning Guide
      https://platform.openai.com/docs/guides/fine-tuning
      (Simplest approach, no infrastructure needed)

- [ ] LoRA/QLoRA concepts
      Hugging Face: PEFT Documentation
      https://huggingface.co/docs/peft
      (Parameter-efficient fine-tuning — understand concept, not deep math)

- [ ] When to use each approach
      Full fine-tuning vs LoRA vs prompt tuning

### 10.3 Data Preparation (~3 hrs)
- [ ] Creating training datasets
      Format requirements, quality over quantity

- [ ] Quality control
      Data validation, deduplication, format checking

- [ ] Evaluation datasets
      Hold-out test sets, evaluation metrics

### Recommended Course (Optional)
- [ ] Udemy: The AI Engineer Course: Complete AI Engineer Bootcamp
      https://www.udemy.com/course/the-ai-engineer-course-complete-ai-engineer-bootcamp/

### Level 10 Checkpoint
- [ ] Can evaluate if fine-tuning is right for the problem
- [ ] Can prepare training data and execute fine-tuning via OpenAI API

---

## 🎯 Capstone Project C: Multi-source Research Agent (~2w)

**Complete after Level 10**

- [ ] **Task:** Build agent answering complex questions by searching internal docs (RAG), calling web search, synthesizing from multiple sources
- [ ] **Tech Stack:** LangGraph + Vector DB + Web Search Tool + MCP + LangFuse for tracing
- [ ] **Skills:** Agents, multi-step reasoning, tool orchestration, observability, evaluation
- [ ] **Deliverable:** Working agent + full observability setup + evaluation report

---

## Time Summary

| Level | Topic | Hours |
|-------|-------|-------|
| 1 | Conceptual Foundation | 2w |
| 2 | Python for AI | 1w-2w |
| 3 | Data & Preprocessing | 2w-3w |
| 4 | Cloud AI Services | 3w |
| — | Capstone A: Receipt Processor | 1w |
| 5 | LLMs & Prompt Engineering | 2w-3w |
| 6 | RAG & Vector Databases | 3w |
| 7 | Building AI Applications | 3w-4w |
| — | Capstone B: HR Policy Chatbot | 3w |
| 8 | Production & Operations | 3w |
| 9 | AI Safety & Ethics | 1w |
| 10 | Fine-tuning (Optional) | 1w-2w |
| — | Capstone C: Research Agent | 2w |
| **Total** | | **~30w** |

**At 5 hrs/week:** ~35 weeks (8-9 months)
**At 7 hrs/week:** ~25 weeks (6 months)

---

## Books for Deep Dive (Optional)

- [ ] "Natural Language Processing with Transformers" — Hugging Face authors (practical, code-heavy)
- [ ] "Designing Machine Learning Systems" — Chip Huyen (production ML systems)
- [ ] "Building LLM Apps" — Valentino Gagliardi (practical LLM development)

---

## Related Documents

- ML Engineer Roadmap — for training custom models (separate document)
- AI Engineer Interview Prep — common questions and system design (separate document)

---

## Weekly Log

| Week | Dates | Notes |
|------|-------|-------|
| 1 | | |