# twiComplete

**Akan N-gram Autocomplete System**

A natural language processing tool for Akan (Twi) language autocomplete, featuring an interactive CLI with trigram backoff smoothing and intelligent word prediction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

twiComplete is an autocomplete system specifically designed for the Akan (Twi) language. It leverages N-gram language modeling with backoff smoothing techniques to provide accurate word predictions in an interactive command-line interface.

**Key Highlights:**

- **Language-Specific**: Optimized for Akan/Twi language patterns
- **Advanced NLP**: Implements trigram backoff with Laplace smoothing
- **Interactive CLI**: Real-time word suggestions
- **Comprehensive Evaluation**: Perplexity-based model assessment

## Features

### Core Functionality

- **Interactive Autocomplete**: Word-by-word suggestions in real-time
- **Top-N Predictions**: Configurable number of suggestions (default: 3)
- **OOV Handling**: Robust handling of out-of-vocabulary words
- **Laplace Smoothing**: Additive smoothing for probability estimation
- **Vocabulary Management**: Closed vocabulary with `<unk>` token handling

## Installation

### Prerequisites

- Python 3
- pip package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone "https://github.com/kcnewman/twiComplete"
   cd twiComplete
   ```

2. **Install dependencies**

   ```bash
   pip install -r dependencies.txt
   ```

3. **Verify installation**
   ```bash
   python main.py --help
   ```

## Usage

### Interactive Mode (Default)

```bash
python main.py
```

**Interactive Commands:**

- Type Akan words one at a time
- `/new` - Start a new sentence
- `exit` or `quit` - Exit the shell

### Single Prediction Mode

```bash
python main.py -i "me pɛ" -k 3
```

**Parameters:**

- `-i, --input`: Input text for prediction
- `-s, --start`: Filter suggestions by prefix (e.g., 'a')
- `-k, --smoothing`: Smoothing factor (default: 1e-5)
- `--topn`: Number of top suggestions (default: 3)

### Training the Model

```bash
python train.py
```

### Evaluating Model Performance

```bash
python evaluate.py
```

## Data

### Data Sources

- **Source**: [NLP GHANA](https://zenodo.org/records/4432117)
- **Verified Data**: High-quality, curated Akan text (1.7MB)
- **Crowdsourced Data**: Community-contributed sentences (35KB)

## Contributing

We welcome contributions to improve twiComplete!

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

**MIT License** - Copyright (c) 2025 Kelvin Newman

This project is open source and available under the MIT License. See the LICENSE file for details.

---
