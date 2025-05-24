import boto3
import json
import os
from flask import Flask, render_template, jsonify, request # Added request

app = Flask(__name__)

# --- Demo Configuration ---
# Joke file configuration
JOKE_FILES = {
    "classic": "jokes_classic.json",
    "tech": "jokes_tech.json",
    "animal": "jokes_animal.json",
    "spanish": "jokes_spanish.json"
}
ACTIVE_JOKE_FILE = "classic"  # Change this to switch between joke collections

def load_jokes_from_file(joke_file_key):
    file_path = JOKE_FILES.get(joke_file_key, JOKE_FILES["classic"])
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading jokes from {file_path}: {e}")
        # Return default jokes as fallback
        return [
            {"id": 1, "text": "Why don't scientists trust atoms? Because they make up everything!"},
            {"id": 2, "text": "My dog used to chase people on a bike a lot. It got so bad, I had to take his darn bike away."}, # Mildly "unsafe"
            {"id": 3, "text": "Why don't scientists trust atoms? Because they make up everything!"}, # Repeat
            {"id": 4, "text": "What do you call a fish with no eyes? Fsh!"}
        ]

# Load jokes based on the configured joke file
JOKES = load_jokes_from_file(ACTIVE_JOKE_FILE)

UNSAFE_KEYWORDS = {"darn", "maldita", "maldito"}
MOCKED_FUNNINESS_SCORES = {
    1: 4,
    2: 1,
    3: 4, # As per script, even repeats get consistent funniness score from LLM Judge
    4: 2
}

# --- Bedrock Configuration ---
# Replace 'us-east-1' with your Bedrock region if different
BEDROCK_REGION = 'us-east-1'
try:
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION)
except Exception as e:
    print(f"Error creating Boto3 Bedrock client. Ensure AWS credentials and region are configured. Error: {e}")
    bedrock_client = None 

MODEL_ID_CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"

# --- Evaluation Logic ---
told_jokes_texts = set()
current_joke_index = 0

def check_novelty(joke_text):
    global told_jokes_texts
    # For the demo, novelty is based on the current session.
    # If current_joke_index is 0, it's the start of a new cycle through jokes, so reset told_jokes.
    # This reset is now primarily handled in the / route.
    if joke_text in told_jokes_texts:
        return False, "Repeat"
    told_jokes_texts.add(joke_text)
    return True, "Novel"

def check_safety(joke_text):
    for keyword in UNSAFE_KEYWORDS:
        if keyword.lower() in joke_text.lower():
            return False, f"Unsafe (Reason: '{keyword}')"
    return True, "Safe"

def get_funniness_score_from_llm(joke_text):
    if not bedrock_client:
        print("Bedrock client not initialized. Returning default score.")
        return 1, "(Bedrock Client Error)"

    prompt = f"""Rate the funniness of this joke on a scale of 1 to 5 (1=not funny, 5=very funny).
Only output the numeric score. Do not add any other text, just the number.
Joke: '{joke_text}'.
Score:"""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "temperature": 0.3,
        "top_p": 0.9
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=MODEL_ID_CLAUDE_HAIKU,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response.get('body').read())
        
        if response_body.get("content") and isinstance(response_body["content"], list):
            text_content = ""
            for block in response_body["content"]:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            
            score_text = text_content.strip()
            if score_text.isdigit() and 1 <= int(score_text) <= 5:
                return int(score_text), "(Direct LLM)"
            else:
                print(f"Warning: LLM returned non-numeric or out-of-range score: '{score_text}' for joke: '{joke_text}'. Defaulting to 1.")
                return 1, "(LLM Format Error)"
        else:
            print(f"Warning: Unexpected LLM response format for joke: '{joke_text}'. Defaulting to 1.")
            return 1, "(LLM Response Error)"

    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return 1, "(Bedrock Invoke Error)"

# --- Flask Routes ---
@app.route('/')
def index():
    global told_jokes_texts, current_joke_index
    told_jokes_texts = set() # Reset novelty for a fresh demo run
    current_joke_index = 0   # Reset joke sequence
    return render_template('index.html')

@app.route('/switch_joke_file/<joke_file_key>', methods=['GET'])
def switch_joke_file(joke_file_key):
    global ACTIVE_JOKE_FILE, JOKES, told_jokes_texts, current_joke_index
    
    if joke_file_key not in JOKE_FILES:
        return jsonify({"status": "error", "message": f"Unknown joke file: {joke_file_key}. Available options: {list(JOKE_FILES.keys())}"})
    
    ACTIVE_JOKE_FILE = joke_file_key
    JOKES = load_jokes_from_file(ACTIVE_JOKE_FILE)
    
    # Reset counters for fresh jokes
    told_jokes_texts = set()
    current_joke_index = 0
    
    return jsonify({
        "status": "success", 
        "message": f"Switched to {joke_file_key} jokes",
        "joke_count": len(JOKES)
    })

@app.route('/available_joke_files', methods=['GET'])
def available_joke_files():
    return jsonify({
        "active_file": ACTIVE_JOKE_FILE,
        "available_files": list(JOKE_FILES.keys())
    })

@app.route('/evaluate_next_joke', methods=['GET'])
def evaluate_next_joke():
    global current_joke_index
    if current_joke_index >= len(JOKES):
        return jsonify({"status": "end_of_jokes", "message": "No more jokes!"})

    joke_data = JOKES[current_joke_index]
    joke_text = joke_data["text"]
    joke_id = joke_data["id"] # Get joke_id for mocked scores

    # Perform novelty and safety evaluations (these are always on)
    is_novel, novelty_status = check_novelty(joke_text)
    is_safe, safety_status = check_safety(joke_text)

    # Determine funniness based on selected method
    funniness_method = request.args.get('funniness_method', 'direct_llm') # Default to direct_llm
    funniness_score = 1 # Default score
    funniness_source_label = ""

    if funniness_method == 'mocked':
        funniness_score = MOCKED_FUNNINESS_SCORES.get(joke_id, 1)
        funniness_source_label = "(Mocked)"
    elif funniness_method == 'direct_llm':
        funniness_score, funniness_source_label = get_funniness_score_from_llm(joke_text)
    elif funniness_method == 'bedrock_eval_simulation':
        # Simulate by calling the LLM directly, but label it as a simulation
        # In a real scenario, this would involve a Bedrock Evaluation Job
        funniness_score, _ = get_funniness_score_from_llm(joke_text) # Original label from func is fine
        funniness_source_label = "(Bedrock Eval Sim.)"
        # You could add a small note: "Note: Simulating Bedrock Eval job with direct LLM call for demo."
    else: # Fallback, should not happen if UI is correct
        funniness_score = 1
        funniness_source_label = "(Unknown Method)"


    current_joke_index += 1

    return jsonify({
        "joke": joke_text,
        "novelty": {"is_novel": is_novel, "status": novelty_status},
        "safety": {"is_safe": is_safe, "status": safety_status},
        "funniness": {
            "score": funniness_score,
            "source": funniness_source_label,
            "method_used": funniness_method # For clarity in UI if needed
        },
        "status": "ok"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)