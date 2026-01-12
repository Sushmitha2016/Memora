
import os
import time
import json
import threading
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import cv2
import face_recognition
import chromadb
from chromadb.utils import embedding_functions
import json
import threading
import time
from datetime import datetime
from playsound import playsound
import traceback
import plyer
import dateutil.parser
import sqlite3
import base64
from datetime import timedelta

# Initialize Flask
app = Flask(__name__)
CORS(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

try:
    from plyer import notification
except ImportError:
    notification = None
    print("Plyer notifications not available")


# Initialize ChromaDB (persistent vector storage)
chroma_client = chromadb.PersistentClient(path=os.path.abspath("./chroma_memory"))
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Check if collection exists or create new
try:
    if "conversation_memory" in chroma_client.list_collections():
        memory_collection = chroma_client.get_collection("conversation_memory", embedding_function=embedder)
        print("Loaded existing memory collection")
    else:
        raise ValueError("Collection not found")
except ValueError:
    print("Collection not found, creating a new one...")
    memory_collection = chroma_client.create_collection(
        name="conversation_memory",
        embedding_function=embedder
    )

# Gemini AI Setup
genai.configure(api_key="AIzaSyAmalNhCYaTNuUDzXaLtVFD_t1Q4Z_ayJ4")
chat_model = genai.GenerativeModel("models/gemini-1.5-flash")

# File paths
FACES_DIR = "faces"
FACE_DETAILS = "details.json"
REMINDERS_FILE = "reminders.json"

def load_reminders():
    """Load reminders from a JSON file with error handling"""
    if os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE, "r") as f:
                data = f.read()
                return json.loads(data) if data else []
        except (json.JSONDecodeError):
            return []  # Return empty if file is corrupted
    return []
#def save_reminders(reminders):
 #   """Save reminders to a JSON file with error handling"""
#  try:
 #       with open(REMINDERS_FILE, "w") as f:
  #          json.dump(reminders, f, indent=4)
   #     print(f"[SAVE] Reminders saved: {reminders}")  
    #except Exception as e:
     #   print(f"[ERROR] Failed to save reminders: {e}")  # Log errors


# Replace playsound with this Windows-specific solution
def play_sound():
    try:
        import winsound
        winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS)
    except Exception as e:
        print(f"Sound fallback error: {e}")
        try:
            # Fallback to beep if everything else fails
            import winsound
            winsound.Beep(1000, 500)  # frequency, duration
        except:
            print("Couldn't play any sound")



def save_reminders_to_json():
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reminders")
    reminders = cursor.fetchall()
    conn.close()

    # Convert to list of dicts
    reminder_list = []
    for reminder in reminders:
        reminder_list.append({
            "id": reminder[0],
            "text": reminder[1],
            "time": reminder[2],
            "date": reminder[3]
        })

    # Save to reminders.json
    with open('reminders.json', 'w') as f:
        json.dump(reminder_list, f, indent=4)

def cleanup_old_reminders():
    """Remove reminders older than 30 days"""
    while True:
        try:
            if os.path.exists('reminders.json'):
                with open('reminders.json', 'r') as f:
                    reminders = json.load(f)
                
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                filtered = [
                    r for r in reminders 
                    if datetime.fromisoformat(r['created_at']) > cutoff
                ]
                
                if len(filtered) != len(reminders):
                    with open('reminders.json', 'w') as f:
                        json.dump(filtered, f, indent=4)
                    print(f"Cleaned up {len(reminders)-len(filtered)} old reminders")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(86400)  # Run once per day


CHAT_HISTORY_FILE = "chat_history.json"

def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(messages, f)

def load_chat_history():
    try:
        if not os.path.exists(CHAT_HISTORY_FILE):
            return []
            
        with open(CHAT_HISTORY_FILE, 'r') as f:
            history = json.load(f)
            # Ensure old format is compatible
            return [{
                "role": msg.get("role"),
                "content": msg.get("content", ""),
                "image": msg.get("image"),
                "timestamp": msg.get("timestamp")
            } for msg in history]
    except:
        return []


# Initialize systems
recognizer = sr.Recognizer()
known_faces_encodings = []
known_faces_names = []
person_details = {}

# ---------------------- Core Functions ----------------------
def store_reminder_in_memory(user_id, reminder_text, reminder_time):
    """Store reminders in both JSON and memory collection"""
    # Store in ChromaDB memory
    memory_collection.add(
        documents=[f"Reminder: {reminder_text} at {reminder_time}"],
        metadatas=[{
            "user_id": user_id,
            "type": "reminder",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }],
        ids=[f"reminder-{time.time()}"]
    )

def load_face_data():
    """Load face data with optimized settings"""
    global known_faces_encodings, known_faces_names, person_details
    
    known_faces_encodings = []
    known_faces_names = []
    
    if os.path.exists(FACES_DIR):
        for filename in os.listdir(FACES_DIR):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    image = face_recognition.load_image_file(os.path.join(FACES_DIR, filename))
                    
                    # Resize image for faster processing
                    small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                    
                    # Single encoding per face with balanced settings
                    encoding = face_recognition.face_encodings(
                        small_image,
                        num_jitters=1,
                        model="small"
                    )[0]  # Just take first encoding
                    
                    known_faces_encodings.append(encoding)
                    known_faces_names.append(os.path.splitext(filename)[0])
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    # Load face details
    try:
        person_details = json.load(open(FACE_DETAILS)) if os.path.exists(FACE_DETAILS) else {}
    except Exception as e:
        print(f"Error loading {FACE_DETAILS}: {e}")
        person_details = {}


def recognize_face_from_webcam():
    """Optimized face recognition with balanced speed/accuracy"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Unable to access camera"

    # Set camera resolution lower for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_name = "Unknown"
    try:
        # Warm up camera
        for _ in range(5): 
            cap.read()
        
        # Single capture with lighting adjustment
        ret, frame = cap.read()
        if not ret:
            return "Unknown"

        # Convert to RGB and resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Faster face detection (hog model is faster than cnn)
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=1,
            model="hog"
        )

        if not face_locations:
            return "Unknown"

        # Get encodings
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, 
            face_locations,
            num_jitters=1,  # Reduced from 3
            model="small"   # Faster model
        )

        # Compare with known faces
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_faces_encodings, 
                encoding,
                tolerance=0.55  # Balanced tolerance
            )
            
            face_distances = face_recognition.face_distance(known_faces_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.55:
                face_name = known_faces_names[best_match_index]
                break

    finally:
        cap.release()
    return face_name
    

def speak(text):
    """Text-to-speech with pyttsx3"""
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    threading.Thread(target=run_tts, daemon=True).start()

# ---------------------- Memory Functions ----------------------

def store_memory(user_id: str, text: str, is_personal: bool = False):
    """Stores conversation context in ChromaDB and avoids duplication"""
    doc_id = f"{user_id}-{datetime.now().timestamp()}"
    memory_collection.add(
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "is_personal": is_personal,
            "timestamp": datetime.now().isoformat(),
            "type": "fact" if not is_personal else "personal"
        }],
        ids=[doc_id]
    )
    print(f"✅ Memory stored: {text} with ID {doc_id}")

def recall_memory(user_id: str, query: str = "", top_k: int = 3) -> list:
    """Retrieves relevant conversation context"""
    try:
        if query:
            results = memory_collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": user_id}
            )
        else:
            results = memory_collection.get(
                where={"user_id": user_id},
                limit=top_k
            )
        
        if results and "documents" in results and results["documents"]:
            print(f"✅ Memory recalled: {results['documents'][0]}")
            return results["documents"][0]
        else:
            print("❌ No memory recalled!")
            return []
    except Exception as e:
        print(f"Memory recall error: {e}")
        return []

def verify_memory_integrity():
    """Check and load memory consistency"""
    try:
        existing_memory = memory_collection.get(limit=5)
        print(f"✅ Memory initialized with {len(existing_memory['ids'])} records after restart.")
    except Exception as e:
        print(f"❌ Memory verification error: {e}")

def get_technical_context(user_id: str, query: str) -> str:
    """Retrieves only factual conversation history with high relevance"""
    results = memory_collection.query(
        query_texts=[query],
        n_results=7,
        where={
            "$and": [
                {"user_id": user_id},
                {"type": "fact"}
            ]
        },
        include=["documents", "distances"]
    )
    
    if not results["documents"]:
        return "No relevant technical context found"
    
    # Filter by similarity score
    relevant = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist < 0.5:  # Only include highly relevant matches
            relevant.append(doc)
    
    return "\n".join(relevant) if relevant else "No sufficiently relevant technical context"

# ---------------------- Response Systems ----------------------

def generate_expert_response(user_id: str, query: str) -> str:
    """Generates strictly fact-based responses from memory"""
    context = get_technical_context(user_id, query)
    
    prompt = f"""You are a factual assistant. Respond ONLY with:
1. Direct quotes from verified conversation history when available
2. "No data available" when information cannot be verified
3. Never speculate beyond the provided context

Relevant Context:
{context}

Query: {query}
Factual Response:"""
    
    response = chat_model.generate_content(prompt)
    return response.text

def generate_conversational_response(user_id: str, query: str) -> str:
    """Generates human-like conversational responses"""
    context = recall_memory(user_id, query)
    prompt = f"You are a friendly and helpful AI assistant. Respond naturally and concisely to the user's query.\n\nPrevious Context:\n{context}\n\nUser: {query}\nAssistant:"
    
    response = chat_model.generate_content(prompt).text
    return response.strip()

# ---------------------- API Endpoints ----------------------

@app.route("/expert-query", methods=["POST"])
def expert_query():
    """Endpoint for technical/factual queries only"""
    data = request.json
    user_input = data.get("message", "").strip()
    user_id = data.get("user_id", "default")

    if not user_input:
        return jsonify({"error": "Empty query"}), 400

    try:
        response = generate_expert_response(user_id, user_input)
        # Store the technical exchange
        store_memory(user_id, f"Technical Query: {user_input}\nResponse: {response}")
        
        return jsonify({
            "response": response,
            "sources": get_technical_context(user_id, user_input),
            "type": "factual"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main conversation endpoint"""
    data = request.json
    user_input = data.get("message", "").strip()
    user_id = data.get("user_id", "default")
    chat_history = load_chat_history()
    if 'messages' in data:
        for msg in data['messages']:
            chat_history.append({
                "role": msg['role'],
                "content": msg['content'],
                "image": msg.get('image'),
                "timestamp": msg.get('timestamp', datetime.now().isoformat())
            })
        save_chat_history(chat_history)
        return jsonify({"status": "success"})
    should_speak = data.get("should_speak", False)
    speech_text = data.get("speech_text", "")
    if not user_input:
        return jsonify({"error": "Empty input"}), 400
    if any(keyword in user_input.lower() for keyword in ["remind", "reminder", "event", "appointment"]):
        # Query memory collection for reminders
        reminder_results = memory_collection.query(
            query_texts=[user_input],
            n_results=3,
            where={
                "$and": [
                    {"user_id": user_id},
                    {"type": "reminder"}
                ]
            }
        )
        
        if reminder_results and reminder_results["documents"]:
            reminder_context = "\n".join(reminder_results["documents"][0])
            response = f"I found these reminders:\n{reminder_context}"
            return jsonify({"response": response})
    if data.get("should_speak", False):
        speak(data.get("speech_text", ""))

    # Handle message saving
    # Face recognition special case
    if user_input.lower() == "who is this":
        name = recognize_face_from_webcam()
        details = person_details.get(name, "I don't recognize this person")
        #res = person_details.get(name, "I don't recognize this person")
        #response = f"This is {name}. {res}"
        #if response != "I don't recognize this person":
        #    speak(response)
        # Capture and save the webcam image temporarily
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Save the captured image temporarily
            temp_dir = "temp_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"{temp_dir}/captured_{timestamp}.jpg"
            cv2.imwrite(temp_filename, frame)
            
            # Convert to base64 for sending to frontend
            with open(temp_filename, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            chat_history.append({
                "role": "ai",
                "content": f"This is {name}. {details}",
                "image": encoded_image,
                "timestamp": datetime.now().isoformat()
            })
            save_chat_history(chat_history)
            if name != "Unknown":
                speak(f"This is {name}. {details}")
            return jsonify({
                "response": f"This is {name}. {details}",
                "image": encoded_image
            })
        return jsonify({
            "response": f"This is {name}. {details}"
        })
    #elif "adding" in user_input.lower() and "to known faces" in user_input.lower():
        # This handles the face upload user message
        #chat_history.append({
         #   "role": "user",
         #   "content": user_input,
         #   "timestamp": datetime.now().isoformat()
        #})
        #save_chat_history(chat_history)
        #chat_history.append({
         #       "role": "ai",
         #       "content": f"{name}. has been added to known faces with details: .{details}",
          #      "image": encoded_image,
           #     "timestamp": datetime.now().isoformat()
        #})
        #save_chat_history(chat_history)
        
        #return jsonify({"status": "acknowledged"})
    else:        
        response = generate_conversational_response(user_id, user_input)
        # Store conversation
        store_memory(user_id, f"User: {user_input}\nAssistant: {response}")
        speak(response)
        # Process and save
        chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        # ... (your existing AI response logic) ...
        chat_history.append({"role": "ai", "content": response, "timestamp": datetime.now().isoformat()})
        save_chat_history(chat_history)
    #speak(response)
    return jsonify({"response": response})

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify({"history": load_chat_history()})


@app.route("/face", methods=["POST"])
def face_api():
    """Recognizes faces from uploaded image or webcam"""
    data = request.json
    if "image_path" in data:  # From file
        image_path = data["image_path"]
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404
        
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if not face_encodings:
                return jsonify({"error": "No face detected"}), 400
                
            distances = face_recognition.face_distance(known_faces_encodings, face_encodings[0])
            if not distances:
                return jsonify({"name": "Unknown", "details": ""})
                
            best_match = np.argmin(distances)
            name = known_faces_names[best_match] if distances[best_match] < 0.6 else "Unknown"
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:  # From webcam
        name = recognize_face_from_webcam()
    
    return jsonify({
        "name": name,
        "details": person_details.get(name, "No information available")
    })

@app.route("/speech", methods=["POST"])
def speech_api():
    """Converts speech to text"""
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return jsonify({"text": text})
        except sr.WaitTimeoutError:
            return jsonify({"error": "No speech detected"}), 408
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/upload-face', methods=['POST'])
def upload_face():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
            
        file = request.files['image']
        name = request.form.get('name', '').strip()
        details = request.form.get('details', '').strip()
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        # Read the file data for returning to frontend
        file_data = file.read()
        encoded_image = base64.b64encode(file_data).decode('utf-8')    
        # Save the image
        filename = f"{name}.{file.filename.split('.')[-1]}"
        filepath = os.path.join(FACES_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        # Update details.json
        if os.path.exists(FACE_DETAILS):
            with open(FACE_DETAILS, 'r') as f:
                face_details = json.load(f)
        else:
            face_details = {}
            
        face_details[name] = details
        
        with open(FACE_DETAILS, 'w') as f:
            json.dump(face_details, f, indent=4)
            
        # Reload face data
        load_face_data()
        confirmation = f"{name} has been added to known faces with details: {details}"
        speak(confirmation)
        
        return jsonify({
            "status": "success",
            "message": "Face added successfully",
            "name": name,
            "details": details,
            "image": encoded_image,
            "speech_text": f"{name} has been added to known faces with details: {details}",
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/who-is-this", methods=["POST"])
def who_is_this():
    try:
        # Single capture with optimized function
        name = recognize_face_from_webcam()
        details = person_details.get(name, "No additional information available")
        
        # Capture image only once for response
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                "response": f"This is {name}. {details}",
                "image": None
            })

        # Save temporary image
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{temp_dir}/captured_{timestamp}.jpg"
        cv2.imwrite(temp_filename, frame)
        
        # Convert to base64
        with open(temp_filename, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return jsonify({
            "response": f"This is {name}. {details}",
            "image": encoded_image,
            "confidence": "high" if name != "Unknown" else "low"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add-reminder', methods=['POST'])
def add_reminder():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        print("Received data:", data)  # Debug
        
        if not data or 'text' not in data or 'date' not in data:
            return jsonify({"error": "Missing text or date"}), 400

        # Parse the date (simplified version)
        try:
            reminder_time = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
            reminder_time = reminder_time.astimezone(timezone.utc)
            now_utc = datetime.now(timezone.utc)
            
            if reminder_time <= now_utc:
                return jsonify({"error": "Reminder time must be in the future"}), 400
        except ValueError as e:
            return jsonify({"error": f"Invalid date: {str(e)}"}), 400

        # Load existing reminders
        reminders = []
        if os.path.exists('reminders.json'):
            try:
                with open('reminders.json', 'r') as f:
                    reminders = json.load(f)
            except json.JSONDecodeError:
                reminders = []

        # Add new reminder
        new_reminder = {
            "id": str(int(time.time())),  # Unique ID
            "text": data['text'],
            "timestamp": reminder_time.isoformat(),  # Must be in ISO format
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_triggered": None
        }
        reminders.append(new_reminder)

        # Save to file
        with open('reminders.json', 'w') as f:
            json.dump(reminders, f, indent=4)

        # ALSO store in memory collection
        store_reminder_in_memory(
            user_id, 
            data['text'], 
            reminder_time.isoformat()
        )
        
        return jsonify({
            "status": "success",
            "message": "Reminder added",
            "reminders": reminders
        })
        
    except Exception as e:
        print("Error:", str(e))  # Debug
        return jsonify({"error": str(e)}), 500

@app.route("/get-reminders", methods=["GET"])
def get_reminders():
    """Fetch all reminders with consistent structure"""
    reminders = load_reminders()
    
    # Ensure each reminder has required fields
    processed_reminders = []
    for reminder in reminders:
        processed = {
            "id": reminder.get("id", str(time.time())),
            "text": reminder.get("text", ""),
            "timestamp": reminder.get("timestamp", ""),
            "date": reminder.get("date", reminder.get("timestamp", "")),  # Fallback
            "created_at": reminder.get("created_at", datetime.now(timezone.utc).isoformat())
        }
        processed_reminders.append(processed)
    
    return jsonify({"reminders": processed_reminders})

@app.route("/delete-reminder", methods=["DELETE"])
def delete_reminder():
    """Delete a reminder by timestamp or text"""
    data = request.json
    timestamp = data.get("timestamp")
    text = data.get("text")

    reminders = load_reminders()
    updated_reminders = [r for r in reminders if r["timestamp"] != timestamp and r["text"].lower() != text.lower()]

    if len(updated_reminders) == len(reminders):
        return jsonify({"error": "Reminder not found"}), 404

    #save_reminders(updated_reminders)
    save_reminders_to_json()
    return jsonify({"message": "Reminder deleted successfully", "reminders": updated_reminders})


@app.route("/update-reminder", methods=["PUT"])
def update_reminder():
    try:
        data = request.json
        user_id = data.get('user_id', 'default')
        print(f"Full request data: {data}")  # Debug
        
        # Load current reminders
        if not os.path.exists(REMINDERS_FILE):
            return jsonify({"error": "Reminders file not found"}), 404
            
        with open(REMINDERS_FILE, 'r') as f:
            reminders = json.load(f)
            print(f"Current reminders: {reminders}")  # Debug

        # Find and update the reminder
        updated = False
        for i, reminder in enumerate(reminders):
            if reminder["id"] == data["orgID"]:
                reminders[i] = {
                    "id": reminder["id"],  # Preserve original ID
                    "text": data["newText"],
                    "timestamp": data["newDate"],
                    "created_at": reminder.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "last_triggered": None
                }
                updated = True
                break
        
        if not updated:
            return jsonify({"error": "Reminder not found"}), 404
        
        # Save back to file
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(reminders, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

        store_reminder_in_memory(
            user_id,
            data['newText'],
            data['newDate']
        )
        print(f"Updated reminders: {reminders}")  # Debug
        return jsonify({
            "status": "success",
            "message": "Reminder updated",
            "reminders": reminders
        })
        
    except Exception as e:
        print(f"Update error: {traceback.format_exc()}")  # Full traceback
        return jsonify({"error": str(e)}), 500

@app.route("/reminder/list", methods=["GET"])
def list_reminders():
    """Fetch all reminders"""
    return jsonify({"reminders": load_reminders()})

def reminder_scheduler():
    """Check reminders every 10 seconds"""
    while True:
        try:
            now = datetime.now(timezone.utc)
            print(f"\n[Scheduler] Checking at {now.strftime('%H:%M:%S')}")
            
            # Load reminders
            with open('reminders.json', 'r') as f:
                reminders = json.load(f)
            
            # Process reminders
            updated = False
            for reminder in reminders:
                try:
                    # Parse reminder time (with error handling)
                    reminder_time = datetime.fromisoformat(reminder['timestamp'].replace('Z', '+00:00'))
                    
                    # Calculate time difference in seconds
                    time_diff = (now - reminder_time).total_seconds()
                    
                    # Debug print
                    print(f"Checking: {reminder['text']} (due: {reminder_time}, diff: {time_diff:.1f}s)")
                    
                    # Trigger conditions:
                    # 1. Current time is past reminder time
                    # 2. Within 10-second trigger window
                    # 3. Not already triggered for this exact time
                    if (time_diff >= 0 and 
                        time_diff <= 10 and 
                        reminder.get('last_triggered') != reminder['timestamp']or 'last_triggered' not in reminder):
                        
                        print(f"==> TRIGGERING: {reminder['text']}")
                        play_sound()
                        if notification:
                            notification.notify(
                                title="Memora Reminder",
                                message=reminder['text'],
                                timeout=10
                            )
                        
                        # Mark as triggered with the EXACT reminder timestamp
                        reminder['last_triggered'] = reminder['timestamp']
                        updated = True
                        
                except Exception as e:
                    print(f"Error processing reminder: {str(e)}")
                    continue
            reminders = [r for r in reminders if 
                        (datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) > now or
                        (r.get('last_triggered') != r['timestamp']))]
            
            # Save updates if any reminders were triggered
            if updated:
                with open('reminders.json', 'w') as f:
                    json.dump(reminders, f, indent=4)
                print("Saved updated reminders to file")
                
        except Exception as e:
            print(f"Scheduler error: {str(e)}")
            traceback.print_exc()
        
        time.sleep(10)

if __name__ == "__main__":
    # Initialize systems
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)
    
    load_face_data()
    
    if not known_faces_encodings:
        print(f"Warning: No faces loaded. Add images to {FACES_DIR}")
    
    # Start Flask
    verify_memory_integrity()
    threading.Thread(target=reminder_scheduler, daemon=True).start()
    # Start scheduler
    if not any(t.name == "reminder_scheduler" for t in threading.enumerate()):
        threading.Thread(
            target=reminder_scheduler,
            name="reminder_scheduler",
            daemon=True
        ).start()

    # Start cleanup
    if not any(t.name == "cleanup_thread" for t in threading.enumerate()):
        threading.Thread(
            target=cleanup_old_reminders,
            name="cleanup_thread",
            daemon=True
        ).start()
    app.run(host="0.0.0.0", port=5000, debug=True)