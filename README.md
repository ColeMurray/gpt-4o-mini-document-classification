# Document Classification with Vision Language Models

This repository contains code for evaluating vision-capable language models (specifically GPT-4o variants) on document classification tasks using the RVL-CDIP dataset. The project focuses on zero-shot classification capabilities using only visual inputs, without relying on OCR or additional text information.

## Project Overview

The main goal is to evaluate how well large vision language models can classify document types (e.g., letters, forms, emails) purely from visual information. The project uses the RVL-CDIP dataset, which contains 16 different document classes.

### Key Features

- Zero-shot document classification using vision language models
- Evaluation framework for balanced sampling from RVL-CDIP test set
- Results analysis and metrics computation
- Support for multiple model variants (GPT-4o and GPT-4o-mini)

## Setup

1. Clone this repository
2. Create and activate a Python virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install python-dotenv Pillow datasets openai
   ```
4. Create a `.env` file with your API credentials (if required)

## Project Structure

- `eval-dataset.py`: Main evaluation script for running classification tests
- `results/`: Directory containing evaluation results

## Usage

To run the evaluation:

```bash
python eval-dataset.py
```

The script will:
1. Load a balanced subset of the RVL-CDIP test set
2. Process images through the specified vision language model
3. Generate classification results and metrics
4. Save results to the `results/` directory

## Results

We evaluated the models on a balanced subset of the RVL-CDIP test set, containing 160 samples (10 samples per class). The evaluation was conducted in a zero-shot setting, where models only received the document image without any additional OCR text or metadata.

### Performance Metrics

For GPT-4o-mini:
- Overall Accuracy: 65%
- Invalid Predictions: 2.5%
- Sample Size: 160 documents (10 per class)

Key findings:
- The model achieved reasonable zero-shot performance without any training on the specific dataset
- Performance varied across document types, with structured documents (like letters) showing higher accuracy
- Only 2.5% of predictions were invalid, showing good adherence to the classification schema

Results are saved in JSON format under the `results/` directory with timestamps, including:
- Per-class accuracy breakdown
- Confusion matrix
- Sample predictions with confidence scores

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.