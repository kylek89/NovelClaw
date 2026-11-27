"""
Dataset loading utility: Load long story/creative text data from public datasets
"""
import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from itertools import islice


class DatasetLoader:
    """Dataset loader: Supports various public datasets"""

    # Recommended datasets
    DATASETS = {
        "storydb": {
            "name": "StoryDB",
            "description": "Multilingual narrative dataset, 42 languages, 500+ stories",
            "url": "https://github.com/allenai/storydb",
            "format": "json",
            "language": "multilingual"
        },
        "bookcorpus": {
            "name": "BookCorpus",
            "description": "11,038 free novels in English",
            "url": "https://github.com/soskek/bookcorpus",
            "format": "txt",
            "language": "english"
        },
    }

    @staticmethod
    def load_from_huggingface(
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        streaming: Optional[bool] = None
    ) -> List[Dict]:
        """
        Load datasets from HuggingFace

        Args:
            split: Dataset split (train, test, etc.)
            max_samples: Maximum number of samples to load

        Returns:
            List of data, where each element contains text and metadata
        """
        try:
            from datasets import load_dataset

            print(f"Loading Hugging Face dataset: {dataset_name}")

            # For very large datasets, prefer streaming to avoid downloading the entire dataset before sampling
            if streaming is None:
                streaming = max_samples is not None

            if streaming:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
                iterator = islice(dataset, max_samples) if max_samples else dataset
            else:
                dataset = load_dataset(dataset_name, split=split)
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                iterator = dataset

            results = []
            for item in iterator:
                # Extract text based on the dataset structure
                text = item.get("text", item.get("content", ""))
                if text:
                    # Clean metadata to prevent the vector database from rejecting lists or complex objects
                    raw_meta = item if isinstance(item, dict) else {}
                    safe_meta = {}
                    for k, v in raw_meta.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            safe_meta[k] = v
                        else:
                            # Convert lists/dicts to strings and truncate
                            try:
                                safe_meta[k] = str(v)[:500]
                            except Exception:
                                pass
                    results.append({
                        "text": text,
                        "title": item.get("title", "Unknown"),
                        "author": item.get("author", "Unknown"),
                        "genre": item.get("genre", "Unknown"),
                        "metadata": safe_meta
                    })

            print(f"Loaded {len(results)} records")
            return results

        except ImportError:
            print("Install the datasets package: pip install datasets")
            return []
        except Exception as e:
            print(f"Could not load dataset: {e}")
            return []

    @staticmethod
    def load_from_directory(
        directory_path: str,
        file_pattern: str = "*.txt",
        encoding: str = "utf-8",
        max_files: Optional[int] = None,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Load text files from a directory

        Args:
            directory_path: Directory path
            file_pattern: File matching pattern
            encoding: File encoding
            max_files: Maximum number of files

        Returns:
            Data list
        """
        import glob

        patterns = file_patterns or [file_pattern]
        files: List[str] = []
        for pat in patterns:
            if recursive:
                files.extend(glob.glob(os.path.join(directory_path, "**", pat), recursive=True))
            else:
                files.extend(glob.glob(os.path.join(directory_path, pat)))

        # Deduplicate and sort for stable import
        files = sorted(set(files))

        if max_files:
            files = files[:max_files]

        results = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                    text = f.read().strip()

                if len(text) > 100:  # Load only files with sufficient content
                    title = Path(file_path).stem
                    results.append({
                        "text": text,
                        "title": title,
                        "author": "Unknown",
                        "genre": "Unknown",
                        "metadata": {"file_path": file_path}
                    })
            except Exception as e:
                print(f"Could not load file {file_path}: {e}")

        print(f"Loaded {len(results)} files from directory")
        return results

    @staticmethod
    def load_from_json(
        json_path: str,
        text_field: str = "text",
        title_field: str = "title"
    ) -> List[Dict]:
        """
        Load data from JSON file

        Args:
            json_path: Path to the JSON file
            text_field: Name of the text field
            title_field: Name of the title field

        Returns:
            Data list
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = []
            if isinstance(data, list):
                for item in data:
                    text = item.get(text_field, "")
                    if text:
                        results.append({
                            "text": text,
                            "title": item.get(title_field, "Unknown"),
                            "author": item.get("author", "Unknown"),
                            "genre": item.get("genre", "Unknown"),
                            "metadata": item
                        })
            elif isinstance(data, dict):
                # If it's a dictionary, try to find a list containing text
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            text = item.get(text_field, "")
                            if text:
                                results.append({
                                    "text": text,
                                    "title": item.get(title_field, "Unknown"),
                                    "author": item.get("author", "Unknown"),
                                    "genre": item.get("genre", "Unknown"),
                                    "metadata": item
                                })

            print(f"Loaded {len(results)} records from JSON")
            return results

        except Exception as e:
            print(f"Loading JSON failed: {e}")
            return []

    @staticmethod
    def download_sample_data(output_dir: str = "./sample_data"):
        """
        Download sample data (if available)

        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Sample data directory: {output_dir}")
        print("\nDataset download options:")
        print("1. HuggingFace: use load_from_huggingface()")
        print("2. After downloading, use load_from_directory()")
        print("\nEnglish datasets:")
        print("- BookCorpus")
        print("- StoryDB")


def quick_load_english_novels(max_samples: int = 100) -> List[Dict]:
    """
    Quickly load English novel data

    Args:
        max_samples: Maximum number of samples

    Returns:
        Data list
    """
    # BookCorpus usually needs to be downloaded from GitHub
    print("Download the English dataset manually")
    print("Recommended: BookCorpus (https://github.com/soskek/bookcorpus)")
    return []

