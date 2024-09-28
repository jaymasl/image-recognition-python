import ollama
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm
import argparse

ITERATIONS_COUNT = 30
TOP_KEYWORDS_COUNT = 10

def extract_keywords_from_image(image_path: str, iterations: int = ITERATIONS_COUNT) -> list:
    """
    Extracts keywords from an image using a chat model.

    Args:
        image_path (str): The path to the image file.
        iterations (int): The number of times to query the model.

    Returns:
        list: A list of the top keywords and their counts.
    """
    replies = []

    for _ in tqdm(range(iterations), desc="Extracting keywords", unit="iteration"):
        try:
            response = ollama.chat(
                model='llava:13b',
                messages=[
                    {
                        'role': 'user',
                        'content': (
                            "List at least fifteen (15) keywords separated by commas."
                            "Terms that visibly describe this image."
                            "Visible descriptive elements."
                            "Important identification information."
                            "Do not write a sentence."
                        ),
                        'images': [image_path]
                    }
                ]
            )
            replies.append(response['message']['content'])
        except Exception as e:
            print(f"Error during chat request: {e}")

    all_words = ', '.join(replies).split(', ')
    word_counts = Counter(word.strip().lower() for word in all_words)

    return word_counts.most_common(TOP_KEYWORDS_COUNT)

def aggregate_keyword_counts(top_words: list) -> list:
    """
    Aggregates keyword counts based on substring matches.

    Args:
        top_words (list): A list of tuples containing keywords and their counts.

    Returns:
        list: A sorted list of aggregated keyword counts.
    """
    aggregated_counts = {}

    for word, count in top_words:
        # Check for existing keywords to aggregate counts
        to_remove = []
        for existing_word in list(aggregated_counts.keys()):
            if word in existing_word or existing_word in word:
                aggregated_counts[existing_word] += count
                if existing_word != word:
                    to_remove.append(existing_word)
                break
        else:
            # If no match was found, add the new word
            aggregated_counts[word] = count
        
        # Remove any keywords that are now redundant
        for existing_word in to_remove:
            del aggregated_counts[existing_word]

    return sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)

def main(image_path: str):
    """
    Main function to extract and aggregate keywords from an image.

    Args:
        image_path (str): The path to the image file.
    """
    top_words = extract_keywords_from_image(image_path)
    aggregated_counts = aggregate_keyword_counts(top_words)

    if aggregated_counts:
        guess_sentence = f"Image likely depicts: {', '.join(word for word, _ in aggregated_counts[:-1])}, and {aggregated_counts[-1][0]}."
        print(guess_sentence)

        print("\nKeyword counts:")
        print(tabulate(aggregated_counts, headers=["Keyword", "Count"], tablefmt="pretty"))
    else:
        print("No keywords extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keywords from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    main(args.image_path)
