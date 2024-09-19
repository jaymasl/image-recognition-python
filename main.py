import ollama
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm  # Import tqdm for progress bar

ITERATIONS = 30

def extract_keywords_from_image(image_path, iterations=ITERATIONS):
    """
    Extracts keywords from an image using a chat model.

    Args:
        image_path (str): The path to the image file.
        iterations (int): The number of times to query the model.

    Returns:
        list: A list of the top 15 keywords and their counts.
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
                            """Give a list of single keywords separated by commas.
                            Give words that visibly describe this image.
                            Accurate contextual visible descriptive elements.
                            Give important identification information.
                            Do not write a sentence."""
                        ),
                        'images': [image_path]
                    }
                ]
            )
            replies.append(response['message']['content'])
        except Exception as e:
            print(f"Error during chat request: {e}")
            return []

    all_words = ', '.join(replies).split(', ')
    word_counts = Counter(word.strip().lower() for word in all_words)

    return word_counts.most_common(10)

def aggregate_keyword_counts(top_words):
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
            if word in existing_word:  # If the new word is a substring of an existing word
                aggregated_counts[existing_word] += count
                break
            elif existing_word in word:  # If the existing word is a substring of the new word
                aggregated_counts[word] = aggregated_counts.get(word, 0) + aggregated_counts[existing_word] + count
                to_remove.append(existing_word)
                break
        else:
            # If no match was found, add the new word
            aggregated_counts[word] = count
        
        # Remove any keywords that are now redundant
        for existing_word in to_remove:
            del aggregated_counts[existing_word]

    return sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)

def main(image_path):
    """
    Main function to extract and aggregate keywords from an image.

    Args:
        image_path (str): The path to the image file.
    """
    # Get keywords for the specified image
    top_words = extract_keywords_from_image(image_path)

    # Aggregate keywords based on substring matches
    aggregated_counts = aggregate_keyword_counts(top_words)

    # Create guess sentence using the aggregated keywords
    if aggregated_counts:
        guess_sentence = f"Image likely depicts: {', '.join(word for word, _ in aggregated_counts[:-1])}, and {aggregated_counts[-1][0]}."
        print(guess_sentence)

        # Print the counts of each keyword in a formatted table
        print("\nKeyword counts:")
        print(tabulate(aggregated_counts, headers=["Keyword", "Count"], tablefmt="pretty"))
    else:
        print("No keywords extracted.")

if __name__ == "__main__":
    main('./image1.jpg')
