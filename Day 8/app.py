import os
import math
from google import genai

# Load API key securely
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not found.")
    exit()

# Initialize the NEW client
client = genai.Client(api_key=api_key)

# A mathematical function to calculate how similar two vectors are.
def calculate_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

# Step 6 & Practical Task: 10 pieces of information about a fictional business
print("Generating embeddings for our database... please wait.")
database_info = [
    "TechNova offers a 12-week intensive Python web development bootcamp.",
    "Our campus is located in downtown San Francisco, near the tech hub.",
    "Students get lifetime access to our online learning platform.",
    "We offer job placement assistance and resume reviews for all graduates.",
    "The tuition fee for the full-time program is $10,000.",
    "Scholarships are available for underrepresented groups in tech.",
    "Instructors at TechNova have at least 5 years of industry experience.",
    "We also offer a part-time data science course that takes 6 months.",
    "Students need to bring their own Mac or Windows laptops to the classes.",
    "Our next cohort starts on the first Monday of September."
]

# FIX: Generate embeddings ONE BY ONE to avoid the SDK batching bug
doc_embeddings = []
for info in database_info:
    response = client.models.embed_content(
        model='gemini-embedding-001', # <--- The official, updated embedding model
        contents=info
    )
    # Extract the numerical vector and add it to our list
    doc_embeddings.append(response.embeddings[0].values)

# Step 7 & Practical Task: Test questions
test_questions = [
    "Where is the school located?",                     
    "How much does it cost to study here?",             
    "Do you help people find work after they finish?",  
]

print("\n--- Semantic Search Results ---")

# Compare each test question to the database
for question in test_questions:
    
    # Turn the user's question into an embedding vector
    question_response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=question
    )
    question_embedding = question_response.embeddings[0].values
    
    best_score = 0
    best_match = ""
    
    # Compare the question vector to every document vector
    for i in range(len(database_info)):
        score = calculate_similarity(question_embedding, doc_embeddings[i])
        
        # Keep track of the highest score
        if score > best_score:
            best_score = score
            best_match = database_info[i]
            
    # Print the results
    print(f"User Question: '{question}'")
    print(f"Best Match Found: '{best_match}'")
    print(f"Similarity Score: {best_score:.4f}\n")