#!/usr/bin/env python3

"""
Evaluate GPT-4o or GPT-4o-mini on a balanced subset of the RVL-CDIP test split
using purely vision-based zero-shot classification.
"""

import os
import random
import base64
import json
import shutil
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv
import logging
from PIL import UnidentifiedImageError
from io import BytesIO

# Install these if missing: pip install openai datasets Pillow python-dotenv
from openai import OpenAI
from datasets import load_dataset
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

print(f"Using key with prefix {os.getenv('OPENAI_API_KEY')[:5]}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of 16 classes in RVL-CDIP
LABEL_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo",
]

@dataclass
class EvalConfig:
    """Configuration for evaluation run"""
    model_name: str = "gpt-4o-mini"
    samples_per_class: int = 10
    temp_dir: str = "temp_images"
    max_retries: int = 3
    batch_size: int = 1
    temperature: float = 0.0
    max_tokens: int = 20
    system_prompt: Optional[str] = None

def is_valid_image(example) -> bool:
    """
    Check if an image in the dataset is valid and can be opened.
    
    Args:
        example: Dataset example containing an image
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        if isinstance(example['image'], Image.Image):
            # If it's already a PIL Image, try to verify it
            example['image'].verify()
            return True
        elif isinstance(example['image'], bytes):
            # If it's bytes, try to open it
            img = Image.open(BytesIO(example['image']))
            img.verify()
            return True
        return False
    except Exception as e:
        logger.debug(f"Invalid image found: {e}")
        return False

def encode_image_to_base64(img_path: str) -> str:
    """
    Load an image from disk and return a base64-encoded string.
    """
    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image {img_path}: {e}")
        raise

def create_balanced_sample(dataset, samples_per_class: int) -> List[int]:
    """
    Create a balanced sample of indices ensuring equal representation of all classes.
    
    Args:
        dataset: The dataset to sample from
        samples_per_class: Number of samples to select for each class
        
    Returns:
        List of selected indices
    """
    # Group valid examples by class
    class_indices: Dict[int, List[int]] = defaultdict(list)
    
    # Iterate through dataset with error handling
    for idx in range(len(dataset)):
        try:
            example = dataset[idx]
            if not is_valid_image(example):
                continue
            class_indices[example['label']].append(idx)
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}")
            continue
    
    # Verify we have enough valid samples for each class
    min_samples = min(len(indices) for indices in class_indices.values())
    if samples_per_class > min_samples:
        logger.warning(f"Requested {samples_per_class} samples per class, but only {min_samples} available for some classes")
        samples_per_class = min_samples
    
    # Sample equally from each class
    balanced_indices = []
    for class_id in range(len(LABEL_NAMES)):
        if class_id in class_indices:
            selected = random.sample(class_indices[class_id], samples_per_class)
            balanced_indices.extend(selected)
    
    return balanced_indices

def cleanup_temp_files(temp_dir: str):
    """Remove temporary image directory and contents"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up {temp_dir}: {e}")

def gpt4o_zero_shot_classify_image(
    base64_img: str,
    candidate_labels: List[str],
    config: EvalConfig,
    attempt: int = 0
) -> Tuple[str, float]:
    """
    Send an image to GPT-4o endpoint for zero-shot classification with retry logic.
    Returns both the predicted label and the API call success status.
    """
    system_text = config.system_prompt or (
        "You are a vision-based document classifier. "
        "Classify the input image into exactly one of the following document types:\n\n"
        + ", ".join(candidate_labels) + "\n\n"
        "Respond with only the document type that best fits the image."
    )

    messages = [
        {
            "role": "system",
            "content": system_text
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the document image to classify:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                        "detail": "auto"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        predicted_label = response.choices[0].message.content.strip().lower()
        return (predicted_label, 1.0) if predicted_label in candidate_labels else ("INVALID", 0.0)
    except Exception as e:
        if attempt < config.max_retries - 1:
            logger.warning(f"API call failed, attempt {attempt + 1}/{config.max_retries}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
            return gpt4o_zero_shot_classify_image(base64_img, candidate_labels, config, attempt + 1)
        logger.error(f"API Error after {config.max_retries} attempts: {e}")
        return ("INVALID", 0.0)

def evaluate_model(dataset, indices: List[int], config: EvalConfig) -> Dict:
    """
    Evaluate the model on the selected indices and return metrics.
    """
    os.makedirs(config.temp_dir, exist_ok=True)
    
    metrics = {
        "correct": 0,
        "total": 0,
        "invalid": 0,
        "class_accuracies": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sample_results": []  # Track individual sample results
    }

    for i, idx in enumerate(indices):
        try:
            example = dataset[idx]
            if not is_valid_image(example):
                logger.warning(f"Skipping invalid image at index {idx}")
                continue
                
            pil_img = example["image"]
            label_id = example["label"]
            true_label = LABEL_NAMES[label_id]

            # Save and process image
            img_path = f"{config.temp_dir}/test_{i}.jpg"
            pil_img.save(img_path, format="JPEG")
            base64_img = encode_image_to_base64(img_path)

            # Get prediction
            predicted_label, api_success = gpt4o_zero_shot_classify_image(
                base64_img=base64_img,
                candidate_labels=LABEL_NAMES,
                config=config
            )

            # Track sample result
            sample_result = {
                "dataset_index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": predicted_label == true_label,
                "invalid": predicted_label == "INVALID",
                "api_success": api_success
            }
            metrics["sample_results"].append(sample_result)

            # Update metrics
            metrics["total"] += 1
            metrics["class_accuracies"][true_label]["total"] += 1
            
            if predicted_label == "INVALID":
                metrics["invalid"] += 1
            if predicted_label == true_label:
                metrics["correct"] += 1
                metrics["class_accuracies"][true_label]["correct"] += 1

            logger.info(f"Sample {i+1}/{len(indices)} | True: {true_label:<20} | Pred: {predicted_label}")

        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            continue

    return metrics

def print_metrics(metrics: Dict):
    """Print detailed evaluation metrics."""
    total = metrics["total"]
    accuracy = metrics["correct"] / total if total > 0 else 0.0
    invalid_pct = 100.0 * metrics["invalid"] / total if total > 0 else 0.0

    print("\n=== Final Report ===")
    print(f"Number of test samples: {total}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Invalid answers: {metrics['invalid']} ({invalid_pct:.2f}%)")
    
    print("\nPer-class Accuracies:")
    for label, stats in metrics["class_accuracies"].items():
        class_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{label:<20}: {class_acc*100:.2f}% ({stats['correct']}/{stats['total']})")

def save_evaluation_results(metrics: Dict, config: EvalConfig, output_dir: str = "results"):
    """Save evaluation results and configuration for reproducibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group sample results by class for easier analysis
    samples_by_class = defaultdict(list)
    for sample in metrics["sample_results"]:
        samples_by_class[sample["true_label"]].append(sample)
    
    results = {
        'metrics': {
            'overall': {
                'total_samples': metrics["total"],
                'correct': metrics["correct"],
                'invalid': metrics["invalid"],
                'accuracy': metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0,
                'invalid_percentage': 100.0 * metrics["invalid"] / metrics["total"] if metrics["total"] > 0 else 0.0
            },
            'per_class': {
                label: {
                    'total': stats["total"],
                    'correct': stats["correct"],
                    'accuracy': stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                    'samples': samples_by_class[label]
                }
                for label, stats in metrics["class_accuracies"].items()
            }
        },
        'config': {
            'model_name': config.model_name,
            'samples_per_class': config.samples_per_class,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'max_retries': config.max_retries
        },
        'timestamp': datetime.now().isoformat(),
        'label_names': LABEL_NAMES
    }
    
    filename = f"eval_results_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")

def main():
    # Create configuration
    config = EvalConfig(
        model_name="gpt-4o-mini",
        samples_per_class=10,
        temp_dir="temp_images"
    )
    
    # Load the RVL-CDIP dataset
    ds = load_dataset("aharley/rvl_cdip")
    test_ds = ds["test"]
    
    # Create balanced sample
    balanced_indices = create_balanced_sample(test_ds, config.samples_per_class)
    
    if not balanced_indices:
        logger.error("No valid samples found in the dataset")
        return
        
    try:
        # Evaluate model
        metrics = evaluate_model(
            test_ds, 
            balanced_indices,
            config
        )
        
        # Print results
        print_metrics(metrics)
        
        # Save results
        save_evaluation_results(metrics, config)
        
    finally:
        # Clean up temporary files
        cleanup_temp_files(config.temp_dir)

if __name__ == "__main__":
    main()