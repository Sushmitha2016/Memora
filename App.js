
import React, { useState, useEffect, useRef } from "react";
import Button from "./components/ui/button";
import Input from "./components/ui/input";
import { Card, CardContent } from "./components/ui/card";
import { Mic, Send, Plus, Edit, Trash, Upload } from "lucide-react";
import "./App.css";
import "./index.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [listening, setListening] = useState(false);
  const [reminders, setReminders] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [reminderText, setReminderText] = useState("");
  const [reminderDate, setReminderDate] = useState("");
  const [editingIndex, setEditingIndex] = useState(null);
  const chatEndRef = useRef(null); // Reference for auto-scroll
  const historyEndRef = useRef(null); // Auto-scroll for history
  const formatForDateTimeLocal = (isoString) => {
  const date = new Date(isoString);
  const localDate = new Date(date.getTime() - (date.getTimezoneOffset() * 60000));
  return localDate.toISOString().slice(0, 16);
};
  const openEditModal = (index) => {
  const reminder = reminders[index];
  console.log("Raw reminder data:", reminder);
  
  // Convert stored UTC to local datetime-local format
  const formattedDate = formatForDateTimeLocal(reminder.timestamp);
  console.log("Formatted for edit:", formattedDate);
  
  setReminderText(reminder.text);
  setReminderDate(formattedDate);
  setEditingIndex(index);
  setShowModal(true);
};
  const [showFaceUploadModal, setShowFaceUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [faceName, setFaceName] = useState("");
  const [faceDetails, setFaceDetails] = useState("");
  const [filePreview, setFilePreview] = useState(null); 
    
    // Auto-scroll to the bottom when new message arrives in right panel
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Auto-scroll for chat history in left panel
  useEffect(() => {
    if (historyEndRef.current) {
      historyEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

    // Load chat history on startup
useEffect(() => {
    const fetchChatHistory = async () => {
        try {
            const response = await fetch('http://localhost:5000/chat-history');
            const data = await response.json();
            setMessages(data.history || []);
        } catch (error) {
            console.error("Failed to load chat history:", error);
        }
    };
    fetchChatHistory();
}, []);

    
  // Function to send user message to backend
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) throw new Error("Server Error");

      const data = await response.json();

      if (data && data.response) {
        setMessages((prev) => [
          ...prev,
          { role: "ai", content: data.response, image: data.image },
        ]);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error("Chatbot Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "I'm having trouble responding right now." },
      ]);
    }
  };

  // Speech Recognition Function
  const handleSpeechRecognition = () => {
    if (!("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.start();
    setListening(true);

    recognition.onresult = (event) => {
      const speechText = event.results[0][0].transcript;
      setInput(speechText);
      setListening(false);
    };

    recognition.onerror = (event) => {
      console.error("Speech Recognition Error:", event.error);
      setListening(false);
    };
  };

  // Handle 'Enter' key press for sending messages
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default behavior
      handleSend();
    }
  };

useEffect(() => {
    fetchReminders();
}, []);
// Add this useEffect for initial load
useEffect(() => {
    const fetchAllReminders = async () => {
        try {
            const response = await fetch('http://localhost:5000/get-reminders');
            const data = await response.json();
            setReminders(data.reminders || []);
        } catch (error) {
            console.error("Failed to load reminders:", error);
        }
    };
    fetchAllReminders();
}, []); 


useEffect(() => {
  console.log("Modal state:", showFaceUploadModal); // Debug modal state
}, [showFaceUploadModal]);
    
const saveReminders = async (reminder) => {
    try {
        const response = await fetch("http://localhost:5000/add-reminder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reminder),
        });

        const data = await response.json();
        console.log("Reminder saved:", data);
        fetchReminders(); // Refresh reminder list after adding
    } catch (error) {
        console.error("Error saving reminder:", error);
    }
};
const fetchReminders = async () => {
    try {
        const response = await fetch("http://localhost:5000/get-reminders");
        if (!response.ok) {
            throw new Error("Failed to fetch reminders");
        }
        const data = await response.json();
        setReminders(data.reminders || []);
    } catch (error) {
        console.error("Error fetching reminders:", error);
    }
};
const handleAddReminder = async () => {
    console.log("Add button clicked"); // Debug 1
    console.log("Current values:", { reminderText, reminderDate }); // Debug 2
    
    if (!reminderText || !reminderDate) {
        console.log("Validation failed"); // Debug 3
        alert("Please fill both fields");
        return;
    }

    try {
        console.log("Preparing to send request"); // Debug 4
        const isoDate = new Date(reminderDate).toISOString();
        console.log("Formatted date:", isoDate); // Debug 5

        const response = await fetch("http://localhost:5000/add-reminder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: reminderText,
                date: isoDate,
                user_id: "current_user_id"
            })
        });
        console.log("Response status:", response.status); // Debug 6

        const data = await response.json();
        console.log("Response data:", data); // Debug 7

        if (!response.ok) {
            throw new Error(data.error || "Failed to add reminder");
        }

        console.log("Success! Closing modal"); // Debug 8
        setReminders(data.reminders);
        setShowModal(false);
        setReminderText("");
        setReminderDate("");
        
    } catch (error) {
        console.error("Full error:", error); // Debug 9
        alert(`Error: ${error.message}`);
    }
};

  // Handle deleting a reminder
  const handleDeleteReminder = async (index) => {
    const reminderToDelete = reminders[index];

    try {
        // Optimistic UI update - remove from local state first
        const updatedReminders = [...reminders];
        updatedReminders.splice(index, 1); // Remove the reminder at this index
        setReminders(updatedReminders);
        const response = await fetch("http://localhost:5000/delete-reminder", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: reminderToDelete.text }),
        });

        if (!response.ok) throw new Error("Failed to delete reminder");

        const data = await response.json();
        setReminders(data.reminders); // Update reminders state with new list
    } catch (error) {
        console.error("Error deleting reminder:", error);
    }
};
const formatDisplayTime = (isoString) => {
  const date = new Date(isoString);
  return date.toLocaleTimeString('en-US', {
    hour12: true,
    hour: '2-digit',
    minute: '2-digit'
  });
};
const handleEditReminder = async () => {
  if (!reminderText || !reminderDate) {
    alert("Please fill both fields");
    return;
  }

  try {
    const updatedReminders = [...reminders];
    const originalReminder = updatedReminders[editingIndex];
    
    // Update local state first for instant UI update
    originalReminder.text = reminderText;
    originalReminder.timestamp = new Date(reminderDate).toISOString();
    setReminders(updatedReminders);

    // Then send to backend
    const response = await fetch("http://localhost:5000/update-reminder", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        orgID: originalReminder.id,
        originalText: originalReminder.text,
        newText: reminderText,
        newDate: originalReminder.timestamp  // Using the ISO string we just created
      })
    });

    if (!response.ok) {
      // Revert if failed
      setReminders(reminders);
      const errorData = await response.json();
      throw new Error(errorData.error || "Update failed");
    }

    // Success - close modal
    setShowModal(false);
    setReminderText("");
    setReminderDate("");
    setEditingIndex(null);
    
  } catch (error) {
    console.error("Update error:", error);
    alert(`Update failed: ${error.message}`);
  }
};
const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (file) {
    setSelectedFile(file);
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setFilePreview(previewUrl);
  }
};

const handleFaceUpload = async () => {
  if (!selectedFile || !faceName) {
    alert("Please select a file and enter a name");
    return;
  }

  try {
    // 1. Convert image to base64 first (sync operation)
    const base64data = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
      reader.readAsDataURL(selectedFile);
    });

    // 2. Prepare form data
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("name", faceName);
    formData.append("details", faceDetails);

    // 3. Show user message immediately
    const userMessage = {
      role: "user",
      content: `Adding ${faceName} to known faces`,
      image: base64data,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    // 4. Upload to backend (await this to complete first)
    const uploadResponse = await fetch("http://localhost:5000/upload-face", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadResponse.json();
    if (!uploadResponse.ok) throw new Error(uploadData.error || "Upload failed");

    // 5. Create AI response
    const aiResponse = `${faceName} has been added to known faces with details: ${faceDetails}`;
    const aiMessage = {
      role: "ai",
      content: aiResponse,
      image: base64data,
      timestamp: new Date().toISOString()
    };

    // 6. Update chat with AI response
    setMessages(prev => [...prev, aiMessage]);

    // 7. Save to history AND trigger speech in one atomic operation
    await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [userMessage, aiMessage],
        should_speak: true,
        speech_text: aiResponse
      }),
    });

    // 8. Immediately close modal and reset
    setShowFaceUploadModal(false);
    setSelectedFile(null);
    setFilePreview(null);
    setFaceName("");
    setFaceDetails("");

  } catch (error) {
    console.error("Upload error:", error);
    alert(`Upload failed: ${error.message}`);
  }
};
const handleWhoIsThis = async () => {
  const userMessage = { role: "user", content: "who is this" };
  setMessages((prev) => [...prev, userMessage]);

  try {
    const response = await fetch("http://localhost:5000/who-is-this", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) throw new Error("Server Error");

    const data = await response.json();
    if (data && data.response) {
      setMessages((prev) => [
        ...prev,
        { 
          role: "ai", 
          content: data.response,
          image: data.image 
        }
      ]);
    }
  } catch (error) {
    console.error("Error:", error);
  }
};/*const handleWhoIsThis = async () => {
  const userMessage = { role: "user", content: "who is this" };
  setMessages((prev) => [...prev, userMessage]);

  try {
    const response = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "who is this" }),
    });

    if (!response.ok) throw new Error("Server Error");

    const data = await response.json();
    if (data && data.response) {
      setMessages((prev) => [
        ...prev,
        { 
          role: "ai", 
          content: data.response,
          image: data.image // Assuming backend sends image data
        }
      ]);
    }
  } catch (error) {
    console.error("Error:", error);
  }
};*/

    
  return (
    <div className="app-container">
      {/* Left Side Panel - Title and Chat History */}
      <div className="left-panel">
        <h1 className="title">Memora - Your AI Assistant</h1>
        <div className="chat-history">
          <h2 className="history-title">Chat History</h2>
          {messages.map((msg, index) => (
            <div
              key={index}
              className={ `${
                msg?.role === "user" ? "user-history" : "ai-history"
              }`}
            >
              {msg?.content || "Error: Missing content"}
            </div>
          ))}
          {/* Auto-scroll reference div for chat history */}
          <div ref={historyEndRef} />
        </div>

        {/* Reminder Modal */}
        {showModal && (
    <div className="modal active">
        <div className="modal-content">
            <h3 style={{marginBottom: '20px',color: 'black'}}>
                {editingIndex !== null ? "Edit Reminder" : "Add Reminder"}
            </h3>
            
            <input
                type="text"
                value={reminderText}
                onChange={(e) => setReminderText(e.target.value)}
                placeholder="Reminder message"
                className="reminder-input"
            />
            
            <input
                type="datetime-local"
                value={reminderDate}
                onChange={(e) => setReminderDate(e.target.value)}
                className="reminder-time-input"
            />
            
            <div className="modal-buttons">
              <button 
                onClick={() => {
                  setShowModal(false);
                  setEditingIndex(null);
                  setReminderText("");
                  setReminderDate("");
                }}
                style={{background: '#f1f1f1', color: '#333'}}
              >
                Cancel
              </button>
              <button 
                onClick={editingIndex !== null ? handleEditReminder : handleAddReminder}
                style={{background: '#4CAF50'}}
              >
                {editingIndex !== null ? "Update" : "Add"}
              </button>
            </div>
        </div>
    </div>
)}
    {/* Add this modal for face upload */}
{showFaceUploadModal && (
  <div className="modal active">
    <div className="modal-content">
      <h3 style={{ color: '#333' }}>Add Known Face</h3>
      
      <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange}
          className="file-input"
          id="face-upload-input"
          style={{ display: 'none' }} // Hide the default input
      />
        <label htmlFor="face-upload-input" className="file-input-label">
          Choose Photo
        </label>
        
        {filePreview && (
          <div className="file-preview">
            <img 
              src={filePreview} 
              alt="Preview" 
              className="preview-image"
            />
            <button 
              onClick={() => {
                setFilePreview(null);
                setSelectedFile(null);
              }}
              className="remove-preview-button"
            >
              Remove
            </button>
          </div>
        )}
      <Input
        value={faceName}
        onChange={(e) => setFaceName(e.target.value)}
        placeholder="Person's name"
      />
      
      <Input
        value={faceDetails}
        onChange={(e) => setFaceDetails(e.target.value)}
        placeholder="Details about this person"
      />
      
      <div className="modal-buttons">
        <button onClick={() => setShowFaceUploadModal(false)}>
          Cancel
        </button>
        <button onClick={handleFaceUpload}>
          Upload
        </button>
      </div>
    </div>
  </div>
)}
        <div className="reminders-container">
          <h2 className="reminders-title">Reminders</h2> {/* Class updated for styling */}
          <ul className="reminders-list">
  {reminders.map((reminder, index) => {
    const date = new Date(reminder.timestamp);

    return (
      <li key={reminder.id || index} className="reminder-item">
        <span>
          {reminder.text} - {formatDisplayTime(reminder.timestamp)}
        </span>
        <div className="reminder-actions">
          <button 
            onClick={() => openEditModal(index)}
            className="icon-button edit-button"
          >
            <Edit size={16} />
          </button>
          <button onClick={() => handleDeleteReminder(index)}
            className="icon-button delete-button"
                >
          <Trash size={16} />
        </button>
        </div>
      </li>
    )
  })}
</ul>
        
          {/* Bigger Plus Button */}
          <Button className="big-add-reminder-button" onClick={() => setShowModal(true)}>
            <Plus size={28} /> {/* Increased size from 20 to 28 */}
          </Button>
        </div>

              
      </div>

      {/* Right Side Panel - Chat Window */}
      <div className="right-panel">
        <Card className="chat-card">
          <CardContent className="chat-content">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={ `${
                  msg?.role === "user" ? "user-message" : "ai-message"
                }`}
              >
                <strong>{msg?.role === "user" ? "You" : "Memora"}:</strong>{" "}
                {msg?.content || "Error: Missing content"}
                {msg?.image && (
                  <img 
                    src={`data:image/jpeg;base64,${msg.image}`} 
                    alt="Attachment" 
                    className="message-image"
                  />
                )}
              </div>
            ))}
            {/* Auto-scroll reference div */}
            <div ref={chatEndRef} />
          </CardContent>
        </Card>

        
        {/* Input Bar - Fixed at Bottom */}
        <div className="input-container">
          <Input
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyDown={handleKeyPress}
          />
          <Button 
            className="ml-2 upload-button"
            onClick={() => setShowFaceUploadModal(true)}
          >
            <Upload size={20} /> {/* You'll need to import Upload from lucide-react */}
          </Button>
        <Button className="ml-2" onClick={handleSend}>
            <Send size={20} />
          </Button>
          <Button
            className="ml-2 mic-button"
            onClick={handleSpeechRecognition}
            disabled={listening}
          >
            <Mic size={20} />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;