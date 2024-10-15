import argparse
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm
import ollama

def extract_keywords(image_path, iterations=30):
    replies = []
    prompt = (
        "List at least fifteen (15) keywords separated by commas."
        "Terms that visibly describe this image."
        "Visible descriptive elements."
        "Important identification information."
        "Do not write a sentence."
    )
    for _ in tqdm(range(iterations), desc="Extracting keywords", unit="iteration"):
        try:
            response = ollama.chat(
                model='llava:13b',
                messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}]
            )
            replies.append(response['message']['content'])
        except Exception as e:
            print(f"Error during chat request: {e}")
    all_words = [word.strip().lower() for reply in replies for word in reply.split(',')]
    return Counter(all_words)

def main():
    parser = argparse.ArgumentParser(description="Extract keywords from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    word_counts = extract_keywords(args.image_path)
    if word_counts:
        top_words = word_counts.most_common(10)
        keywords = [word for word, _ in top_words]
        guess_sentence = f"Image likely depicts: {', '.join(keywords[:-1])}, and {keywords[-1]}."
        print(guess_sentence)
        print("\nKeyword counts:")
        print(tabulate(top_words, headers=["Keyword", "Count"], tablefmt="pretty"))
    else:
        print("No keywords extracted.")

if __name__ == "__main__":
    main()
