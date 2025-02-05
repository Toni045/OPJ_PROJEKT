# Tourism Guide Question Answering System

This project implements a question-answering system for tourism guides using Natural Language Processing (NLP) techniques. 
It extracts meaningful questions and answers from tourism guide texts to provide informative responses. 
The project leverages state-of-the-art transformer models for tokenization and fine-tuning.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)

## Requirements

The project requires the following Python packages:

- `torch` (for deep learning models)
- `transformers` (for transformer models from Hugging Face)
- `nltk` (for text tokenization)
- `pandas` (for data manipulation)
- `scikit-learn` (for model evaluation)
- `evaluate` (for model evaluation metrics)

You can find the complete list of dependencies in the `requirements.txt` file.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Toni045/OPJ_PROJEKT
cd tourism-guide-question-answering
```

### 2. Set up a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
.\venv\Scripts\activate   # For Windows
```

### 3. Install dependencies

```bash
chmod +x install_packages.sh
./install_packages.sh
```

## Setup

1. Ensure you have a tourism_guides.csv file containing tourism guide data. 
This file should include the context column, which contains the tourism guide information (e.g., descriptions of cities, parks, historical sites).

2. The project will prepare the data using the DataPreparator class by creating question-answer pairs based on the provided text.

## Usage

Once the dependencies are installed, you can run the main script to train and evaluate the model:

```bash
python3 main.py
```

## Contributing

- Colarić Borna
- Serezlija Toni