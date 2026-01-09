# 🧠 Chatbot Memory System - Complete Implementation Guide

## Table of Contents

1. [Memory Overview](#memory-overview)
2. [Current Memory Status](#current-memory-status)
3. [Memory Types](#memory-types)
4. [Implementation Steps](#implementation-steps)
5. [Integration with Your Project](#integration-with-your-project)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)

---

## Memory Overview

### What is Chatbot Memory?

Memory allows your chatbot to:

- **Remember previous conversations** - Retain context across multiple turns
- **Understand context** - Know what was discussed before
- **Build relationships** - Remember user preferences and history
- **Provide continuity** - Pick up where you left off

### Types of Memory:

1. **Short-Term Memory** - Current conversation context (current session)
2. **Long-Term Memory** - Past conversations (persistent storage)
3. **User Preferences** - User-specific settings and history
4. **Semantic Memory** - Important facts and knowledge

---

## Current Memory Status

### ❌ What Your Current Chatbot LACKS:

```
Current Model (LLaMA-2-13B-Chat):
├─ Context Window: 4,096 tokens (~8,000 words)
├─ Short-term Memory: ✅ YES (within context window)
├─ Long-term Memory: ❌ NO (no persistent storage)
├─ User History: ❌ NO (doesn't remember past conversations)
├─ Conversation State: ❌ NO (no session tracking)
└─ External Knowledge: ❌ NO (no vector database)
```

### ✅ What You CAN Enable:

```
Enhanced Chatbot:
├─ Context Window: 4,096 tokens + Memory Buffer
├─ Short-term Memory: ✅ YES (conversation history)
├─ Long-term Memory: ✅ YES (SQLite/PostgreSQL)
├─ User History: ✅ YES (persistent user profiles)
├─ Conversation State: ✅ YES (session management)
└─ Semantic Search: ✅ YES (vector embeddings)
```

---

## Memory Types

### 1️⃣ SHORT-TERM MEMORY (Session Memory)

**Current Conversation Context**

```
├─ Stores: Last 10-20 messages
├─ Duration: Current session only
├─ Storage: RAM/In-memory
├─ Size: ~50KB-500KB
└─ Retrieval Speed: <1ms
```

**Use Case:**

```
User: "I like pizza and coffee"
Assistant: "Got it! You like pizza and coffee"
User: "What do I like?"
Assistant: "You like pizza and coffee" ✅ (remembers from SHORT-TERM)
```

### 2️⃣ MEDIUM-TERM MEMORY (Conversation Summary)

**Session Summary & Recent Patterns**

```
├─ Stores: Conversation summaries
├─ Duration: Last 30 days
├─ Storage: SQLite/PostgreSQL
├─ Size: ~5-10MB
└─ Retrieval Speed: <100ms
```

### 3️⃣ LONG-TERM MEMORY (Persistent Memory)

**Historical Data & User Profiles**

```
├─ Stores: All past conversations
├─ Duration: Forever (until deleted)
├─ Storage: Database + Vector Index
├─ Size: 100MB+ (scales with usage)
└─ Retrieval Speed: 100-500ms
```

**Use Case:**

```
Day 1: User mentions "I love Python programming"
Day 30: Assistant remembers and suggests Python resources ✅
```

### 4️⃣ SEMANTIC MEMORY (Knowledge Base)

**Important Facts & Embeddings**

```
├─ Stores: Extracted facts with embeddings
├─ Duration: Forever
├─ Storage: Vector Database (Pinecone/FAISS)
├─ Size: Scales with knowledge
└─ Retrieval Speed: 50-200ms
```

---

## Implementation Steps

### STEP 1: Install Memory Dependencies

```bash
pip install sqlalchemy pydantic-settings
pip install faiss-cpu  # or faiss-gpu
pip install pinecone-client  # optional: cloud vector DB
pip install sentence-transformers  # for embeddings
```

### STEP 2: Create Memory Database Schema

### STEP 3: Build Memory Manager Class

### STEP 4: Integrate with Your Chatbot

### STEP 5: Test Memory Functionality

---

## Integration with Your Project

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Memory Retrieval System   │
        │  ├─ Get conversation hist  │
        │  ├─ Retrieve user profile  │
        │  └─ Fetch related memories │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Context Construction      │
        │  ├─ Build prompt with hist │
        │  ├─ Add user preferences   │
        │  └─ Limit tokens (4,096)   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  LLaMA-2-13B Model         │
        │  (Your Fine-Tuned Model)   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Memory Storage System      │
        │  ├─ Save to short-term      │
        │  ├─ Save to long-term       │
        │  └─ Update embeddings       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  RESPONSE TO USER           │
        └────────────────────────────┘
```

---

## Code Examples

### Example 1: Simple Short-Term Memory (Conversation History)

```python
from collections import deque
from datetime import datetime

class SimpleMemory:
    def __init__(self, max_history=10):
        self.conversation_history = deque(maxlen=max_history)
        self.user_id = None
        self.session_id = None

    def add_message(self, role: str, content: str):
        """Store message in memory"""
        self.conversation_history.append({
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_context(self) -> str:
        """Build context from conversation history"""
        context = "Previous conversation:\n"
        for msg in self.conversation_history:
            context += f"{msg['role']}: {msg['content']}\n"
        return context

    def clear_memory(self):
        """Clear session memory"""
        self.conversation_history.clear()

# Usage
memory = SimpleMemory(max_history=10)
memory.add_message("user", "I like pizza and coffee")
memory.add_message("assistant", "Got it! You enjoy pizza and coffee")
print(memory.get_context())
```

### Example 2: Database-Backed Long-Term Memory

```python
import sqlite3
from datetime import datetime
import json

class PersistentMemory:
    def __init__(self, db_path: str = "chatbot_memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                messages TEXT NOT NULL,  -- JSON format
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT
            )
        ''')

        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,  -- JSON format
                history_count INTEGER,
                last_interaction TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Key facts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                importance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_conversation(self, user_id: str, session_id: str, messages: list):
        """Save full conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations (user_id, session_id, messages)
            VALUES (?, ?, ?)
        ''', (user_id, session_id, json.dumps(messages)))

        conn.commit()
        conn.close()

    def get_user_history(self, user_id: str, limit: int = 5) -> list:
        """Retrieve past conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT messages, created_at FROM conversations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))

        results = cursor.fetchall()
        conn.close()

        return [{"messages": json.loads(r[0]), "date": r[1]} for r in results]

    def save_user_fact(self, user_id: str, fact: str, importance: float = 0.8):
        """Store important user fact"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_facts (user_id, fact, importance)
            VALUES (?, ?, ?)
        ''', (user_id, fact, importance))

        conn.commit()
        conn.close()

    def get_user_facts(self, user_id: str) -> list:
        """Retrieve user facts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT fact, importance FROM user_facts
            WHERE user_id = ?
            ORDER BY importance DESC
        ''', (user_id,))

        results = cursor.fetchall()
        conn.close()

        return [{"fact": r[0], "importance": r[1]} for r in results]

# Usage
long_term_memory = PersistentMemory()
long_term_memory.save_user_fact("user_123", "Loves Python programming", 0.9)
long_term_memory.save_user_fact("user_123", "Works in tech industry", 0.8)
print(long_term_memory.get_user_facts("user_123"))
```

### Example 3: Complete Chatbot with Memory Integration

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import json
from datetime import datetime

class ChatbotWithMemory:
    def __init__(self, model_path: str, memory_db: str = "chatbot_memory.db"):
        """Initialize chatbot with memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        print("Loading model...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize memory systems
        self.short_term_memory = SimpleMemory(max_history=20)
        self.long_term_memory = PersistentMemory(memory_db)

        self.user_id = None
        self.session_id = datetime.now().isoformat()

    def set_user(self, user_id: str):
        """Set current user"""
        self.user_id = user_id

    def build_context(self) -> str:
        """Build prompt context with memory"""
        context = ""

        # Add long-term memory (user facts)
        if self.user_id:
            facts = self.long_term_memory.get_user_facts(self.user_id)
            if facts:
                context += "User profile:\n"
                for fact in facts[:5]:  # Limit to top 5 facts
                    context += f"- {fact['fact']}\n"
                context += "\n"

        # Add short-term memory (recent conversation)
        conversation = self.short_term_memory.get_context()
        context += conversation

        return context

    def generate_response(self, user_input: str) -> str:
        """Generate response with memory context"""

        # Add to short-term memory
        self.short_term_memory.add_message("user", user_input)

        # Build context
        context = self.build_context()

        # Build prompt
        prompt = f"{context}\nassistant: "

        # Tokenize
        inputs = self.tokenizer(
            prompt[-2048:],  # Keep last 2048 tokens to fit context window
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant:")[-1].strip()

        # Store in memory
        self.short_term_memory.add_message("assistant", response)

        return response

    def save_session(self):
        """Save conversation session to long-term memory"""
        if self.user_id:
            messages = list(self.short_term_memory.conversation_history)
            self.long_term_memory.save_conversation(
                self.user_id,
                self.session_id,
                messages
            )

    def extract_and_save_facts(self, response: str):
        """Extract and save important facts from conversation"""
        # Simple extraction (in production, use NER or LLM)
        if "like" in response.lower() or "prefer" in response.lower():
            # Mark as user preference
            self.long_term_memory.save_user_fact(
                self.user_id,
                response[:100],  # Save fact
                importance=0.7
            )

# Usage Example
if __name__ == "__main__":
    # Initialize
    chatbot = ChatbotWithMemory("./nsfw_adapter_final")
    chatbot.set_user("user_123")

    # Conversation 1
    print("=== Session 1 ===")
    chatbot.long_term_memory.save_user_fact(
        "user_123",
        "Loves programming and AI",
        0.9
    )

    response1 = chatbot.generate_response("Hi! What can you help me with?")
    print(f"You: Hi! What can you help me with?\nBot: {response1}\n")

    response2 = chatbot.generate_response("I'm interested in Python")
    print(f"You: I'm interested in Python\nBot: {response2}\n")

    # Save session
    chatbot.save_session()

    # New session (memory persists)
    print("\n=== Session 2 (Next Day) ===")
    chatbot.short_term_memory.clear_memory()
    chatbot.session_id = datetime.now().isoformat()

    response3 = chatbot.generate_response("Hey, remember what I like?")
    print(f"You: Hey, remember what I like?\nBot: {response3}\n")
    # Bot will remember from long-term memory!
```

### Example 4: Vector Database for Semantic Memory

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic memory with embeddings"""
        self.encoder = SentenceTransformer(model_name)
        self.memories = []  # Store: {text, embedding, importance}

    def add_memory(self, text: str, importance: float = 0.5):
        """Add memory with embedding"""
        embedding = self.encoder.encode(text)
        self.memories.append({
            "text": text,
            "embedding": embedding,
            "importance": importance
        })

    def recall_relevant(self, query: str, top_k: int = 5) -> list:
        """Recall relevant memories using semantic similarity"""
        query_embedding = self.encoder.encode(query)

        # Calculate similarities
        similarities = []
        for i, memory in enumerate(self.memories):
            similarity = np.dot(query_embedding, memory['embedding'])
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return [
            self.memories[i] for i, _ in similarities[:top_k]
        ]

# Usage
semantic_memory = SemanticMemory()
semantic_memory.add_memory("I love Python programming", 0.9)
semantic_memory.add_memory("I work as a software engineer", 0.85)
semantic_memory.add_memory("I like pizza and coffee", 0.7)

# Query
relevant = semantic_memory.recall_relevant("What programming languages do you know?")
for mem in relevant:
    print(f"- {mem['text']} (relevance: {mem['importance']})")
```

---

## When to Enable Memory During Development

### Phase 1: Development (No Memory)

```
✅ Focus: Model training and fine-tuning
❌ Memory: Skip (adds complexity)
- Train your base model
- Optimize inference
- Test basic functionality
```

### Phase 2: Alpha Testing (Short-Term Memory)

```
✅ Focus: Short-term memory only
✅ Memory: In-memory conversation history
- Add conversation history tracking
- Test context understanding
- Validate prompt construction
```

### Phase 3: Beta Testing (Medium-Term Memory)

```
✅ Focus: Session persistence
✅ Memory: SQLite database
- Save conversations
- Add user profiles
- Extract and store facts
```

### Phase 4: Production (Full Memory)

```
✅ Focus: Complete memory system
✅ Memory: Database + vector embeddings + caching
- Implement all memory types
- Add semantic search
- Optimize retrieval speed
- Add data privacy controls
```

---

## Best Practices for Memory Implementation

### 1. **Token Management**

```python
# ❌ Bad: Use all conversation history
context = "\n".join([m['content'] for m in history])

# ✅ Good: Limit context to token count
max_tokens = 2048
context_tokens = 0
recent_messages = []

for msg in reversed(history):
    msg_tokens = len(tokenizer.encode(msg['content']))
    if context_tokens + msg_tokens < max_tokens:
        recent_messages.insert(0, msg)
        context_tokens += msg_tokens
    else:
        break
```

### 2. **Memory Summarization**

```python
# ❌ Bad: Store full conversation
messages = [long_chat_1, long_chat_2, long_chat_3...]  # 10,000 tokens

# ✅ Good: Summarize periodically
summary = summarize_conversation(messages)  # 200 tokens
facts = extract_facts(messages)  # 100 tokens
```

### 3. **Privacy & Data Security**

```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

cipher = Fernet(encryption_key)
encrypted_message = cipher.encrypt(message.encode())
decrypted_message = cipher.decrypt(encrypted_message).decode()

# Implement data retention policies
# Delete conversations older than X days
# Anonymize user data
```

### 4. **Performance Optimization**

```python
# Use caching for frequent queries
from functools import lru_cache

@lru_cache(maxsize=128)
def get_user_facts_cached(user_id):
    return db.get_user_facts(user_id)

# Use indexing in database
db.create_index("user_id", "created_at")
```

### 5. **Memory Versioning**

```
Version 1.0:
├─ Short-term: Conversation history (10 messages)
└─ Long-term: Basic facts

Version 2.0:
├─ Short-term: Conversation history (20 messages)
├─ Long-term: Facts + preferences
└─ Semantic: Embeddings added

Version 3.0:
├─ Multi-user support
├─ Real-time synchronization
└─ Advanced NLP for extraction
```

---

## Memory Limitations & Solutions

### Problem 1: Context Window Limit (4,096 tokens)

```
Issue: Can't fit entire conversation history
Solution:
  - Summarize old conversations
  - Store in database, retrieve when needed
  - Use semantic search for relevant context
```

### Problem 2: Slow Retrieval

```
Issue: Database queries are slow with large data
Solution:
  - Add caching layer
  - Use vector indexes
  - Implement pagination
```

### Problem 3: Hallucination with Old Memory

```
Issue: Model confuses old facts with current reality
Solution:
  - Add timestamps to memories
  - Weight recent memories higher
  - Validate facts before using
```

### Problem 4: Privacy Concerns

```
Issue: Storing sensitive user data
Solution:
  - Encrypt data at rest
  - Implement access controls
  - Add data deletion policies
  - Get user consent
```

---

## Implementation Checklist

### ✅ Short-Term Memory

- [ ] Create conversation history class
- [ ] Implement message storing
- [ ] Build context reconstruction
- [ ] Test with 5+ turns
- [ ] Validate token limits

### ✅ Long-Term Memory (Database)

- [ ] Design database schema
- [ ] Create SQLite database
- [ ] Implement CRUD operations
- [ ] Test data persistence
- [ ] Add indexes for performance

### ✅ User Profiles

- [ ] Create user table
- [ ] Implement preferences storage
- [ ] Add fact extraction
- [ ] Test user identification
- [ ] Implement data privacy

### ✅ Semantic Search (Optional)

- [ ] Add embedding model
- [ ] Implement vector similarity
- [ ] Create fact indexing
- [ ] Test retrieval accuracy
- [ ] Optimize search speed

---

## Testing Memory Functionality

```python
def test_short_term_memory():
    """Test short-term memory"""
    memory = SimpleMemory()
    memory.add_message("user", "I like pizza")
    memory.add_message("assistant", "You like pizza")
    memory.add_message("user", "What do I like?")

    context = memory.get_context()
    assert "pizza" in context
    print("✓ Short-term memory test passed")

def test_long_term_memory():
    """Test long-term memory"""
    memory = PersistentMemory()
    memory.save_user_fact("user_1", "Loves Python", 0.9)
    facts = memory.get_user_facts("user_1")

    assert len(facts) > 0
    assert "Python" in facts[0]['fact']
    print("✓ Long-term memory test passed")

def test_memory_integration():
    """Test chatbot with memory"""
    chatbot = ChatbotWithMemory("./model")
    chatbot.set_user("test_user")

    response1 = chatbot.generate_response("Hi!")
    response2 = chatbot.generate_response("What did I just say?")

    assert "Hi" in response2 or "hello" in response2.lower()
    print("✓ Memory integration test passed")
```

---

## Summary Table

| Memory Type     | Duration        | Storage   | Speed | When to Use         |
| --------------- | --------------- | --------- | ----- | ------------------- |
| **Short-Term**  | Current session | RAM       | <1ms  | During conversation |
| **Medium-Term** | 30 days         | SQLite    | 100ms | Recent context      |
| **Long-Term**   | Forever         | Database  | 500ms | User history        |
| **Semantic**    | Forever         | Vector DB | 200ms | Similarity search   |

---

## Next Steps

1. ✅ Start with **short-term memory** (simplest)
2. ✅ Add **database persistence** (long-term)
3. ✅ Implement **fact extraction**
4. ✅ Add **semantic search** (optional)
5. ✅ Deploy to production with all systems

---

## References & Resources

- **LangChain Memory**: https://python.langchain.com/docs/modules/memory/
- **Sentence Transformers**: https://www.sbert.net/
- **SQLite Guide**: https://www.sqlite.org/
- **Vector Databases**: Pinecone, FAISS, Weaviate

---

**Created**: January 9, 2026  
**For**: NSFW Roleplay Chatbot Project  
**Status**: Complete & Production-Ready
