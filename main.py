import ollama
from collections import Counter

def get_keywords(image_path, iterations=35):
    # Initialize an empty list to store the replies
    replies = []

    # Run the chat request for the specified number of iterations
    for _ in range(iterations):
        res = ollama.chat(
            model='llava:13b',
            messages=[
                {'role': 'user',
                'content': '''Only give a concise short list of single keywords.
                Separated by commas, give words that visibly describe this image. 
                Accurate visible descriptive elements.
                Give important identification information.
                Do not write a sentence.''',
                'images': [image_path]}
            ]
        )
        
        # Append the reply to the list
        replies.append(res['message']['content'])

    # Combine all replies into a single string and split into words
    all_words = ', '.join(replies).split(', ')

    # Convert all words to lowercase for case-insensitive counting
    all_words = [word.strip().lower() for word in all_words]

    # Count the occurrences of each word
    word_counts = Counter(all_words)

    # Get the top 10 most common single words
    top_words = word_counts.most_common(10)

    return top_words

def aggregate_keywords(top_words):
    # Create a dictionary to hold aggregated counts
    aggregated_counts = {}

    for word, count in top_words:
        # Check if the word is a substring of any existing key
        found = False
        for existing_word in list(aggregated_counts.keys()):
            if word in existing_word:
                # If the current word is a substring of an existing word, add its count
                aggregated_counts[existing_word] += count
                found = True
                break
            elif existing_word in word:
                # If the existing word is a substring of the current word, add its count
                aggregated_counts[word] = aggregated_counts.get(word, 0) + aggregated_counts[existing_word]
                del aggregated_counts[existing_word]
                found = True
                break
        
        # If not found, add the word to the aggregated counts
        if not found:
            aggregated_counts[word] = count

    return aggregated_counts

# Get keywords for image1 only
top_words_image1 = get_keywords('./image1.jpg')

# Aggregate keywords based on substring matches
aggregated_counts_image1 = aggregate_keywords(top_words_image1)

# Create guess sentence using the aggregated keywords
guess_sentence_image1 = f"Image 1 likely depicts: {', '.join(aggregated_counts_image1.keys())}."

# Print the guess sentence
print(guess_sentence_image1)

# Print the counts of each keyword in a single line for image1
print("\nKeyword counts for Image 1: ", end="")
print(", ".join(f"{word}: {count} times" for word, count in aggregated_counts_image1.items()))
