import csv
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set to store generated questions to avoid duplicates
generated_questions_set = set()

# Function to generate true/false questions using GPT-2 model
def generate_true_false_question(parameters):
    input_text = f"Is {parameters} true or false? Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate question using the model
    output = model.generate(
        input_ids,
        max_length=20,  # Adjust the max_length as needed
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated question
    generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

    # Generate a random True/False answer
    answer = random.choice([True, False])

    return generated_question, answer

# Function to search and generate true/false questions for all parameters
def search_and_generate_true_false_questions(parameters, total_num_questions):
    all_questions = []

    # Combine all parameters into one string
    combined_parameters = " ".join(parameters)

    # Generate true/false questions based on the combined parameters
    for _ in range(total_num_questions):
        generated_question, answer = generate_true_false_question(combined_parameters)
        
        # Check for duplicates
        while generated_question in generated_questions_set:
            generated_question, answer = generate_true_false_question(combined_parameters)
        
        # Add the question to the set
        generated_questions_set.add(generated_question)
        # Add the question to the list to prevent duplicate generation in the next iterations
        all_questions.append((generated_question, answer))

    return all_questions

# Example usage
user_parameters = [
    "Real Madrid",
    "Cristiano Ronaldo",
    "Spain",
    "Champions League",
    "2020"
]

# Specify the total number of questions needed
total_num_questions = 5

# Generate true/false questions for all parameters
all_generated_questions = search_and_generate_true_false_questions(user_parameters, total_num_questions)

# Save the questions to a CSV file
csv_filename = "generated_questions.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write questions to the CSV file
    for question, answer in all_generated_questions:
        csv_writer.writerow([question, answer])

print(f"Generated questions saved to {csv_filename}")
