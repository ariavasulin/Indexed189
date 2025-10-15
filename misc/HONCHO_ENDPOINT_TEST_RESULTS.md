# Honcho MCP Endpoint Test Results
**Tested:** October 15, 2025  
**Server:** https://mcp.honcho.dev  
**Package:** mcp-remote v0.1.29

## Executive Summary
Out of 24 total endpoints, **20 are working** and **4 are broken** due to server-side bugs.

---

## ✅ WORKING ENDPOINTS (20)

### Session Management
1. ✅ **`create_session`** - Creates sessions with optional config
   - Returns: `{"session_id":"...", "config":{...}}`
   
2. ✅ **`list_sessions`** - Lists all sessions
   - Returns: `[{"id":"session_001"}, ...]`
   
3. ✅ **`get_session_metadata`** - Gets session metadata
   - Returns: `{metadata object}`
   
4. ✅ **`set_session_metadata`** - Sets session metadata
   - Returns: "Session metadata set successfully"
   
5. ✅ **`get_session_messages`** - Retrieves messages from session
   - Returns: Array of message objects with id, content, peer_id, metadata, created_at
   
6. ✅ **`get_session_context`** - Gets optimized context within token limit
   - Returns: `{"session_id":"...", "summary":"...", "messages":[...]}`
   
7. ✅ **`get_session_peers`** - Gets all peer IDs in session
   - Returns: `["peer1", "peer2"]`

### Peer Management
8. ✅ **`create_peer`** - Creates peers with optional config
   - Returns: `{"peer_id":"...", "config":{...}}`
   
9. ✅ **`list_peers`** - Lists all peers
   - Returns: `[{"id":"peer1"}, ...]`
   
10. ✅ **`get_peer_metadata`** - Gets peer metadata
    - Returns: `{metadata object}`
    
11. ✅ **`set_peer_metadata`** - Sets peer metadata
    - Returns: "Peer metadata set successfully"

### Session-Peer Operations
12. ✅ **`add_peers_to_session`** - Adds peers to session
    - Returns: "Peers added to session successfully"
    
13. ✅ **`remove_peers_from_session`** - Removes peers from session
    - Returns: "Peers removed from session successfully"

### Message Operations
14. ✅ **`add_messages_to_session`** - Stores messages (uses peer_id)
    - Format: `[{"peer_id":"...", "content":"...", "metadata":{...}}]`
    - Returns: "Messages added to session successfully"

### Search Operations
15. ✅ **`search_session_messages`** - Searches messages in specific session
    - Returns: Array of matching messages (or empty array)
    
16. ✅ **`search_peer_messages`** - Searches messages sent by peer
    - Returns: Array of matching messages (or empty array)

### Workspace Operations
17. ✅ **`get_workspace_metadata`** - Gets workspace metadata
    - Returns: `{metadata object}`
    
18. ✅ **`set_workspace_metadata`** - Sets workspace metadata
    - Returns: "Workspace metadata set successfully"

### AI/Representation Operations
19. ✅ **`get_working_representation`** - Gets peer's working representation in session
    - Returns: `{"representation":{"explicit":[...], "deductive":[...]}}`
    - Provides structured understanding of what peer knows
    
20. ✅ **`chat`** - Query peer's representation with natural language
    - Returns: AI-generated natural language response based on peer's knowledge
    - **Works beautifully!** Provides detailed, contextual answers

---

## ❌ BROKEN ENDPOINTS (4)

### 1. ❌ `start_conversation`
- **Error:** `404 {"detail":"Session not found"}`
- **Issue:** Tries to look up session before creating it
- **Workaround:** Use `create_session` instead

### 2. ❌ `add_turn`
- **Error:** `500 {"detail":"An unexpected error occurred"}`
- **Issue:** Server-side error, unclear cause
- **Workaround:** Use `add_messages_to_session` instead (with peer_id)

### 3. ❌ `get_personalization_insights`
- **Error:** `500 {"detail":"An unexpected error occurred"}`
- **Issue:** Server-side error, personalization broken
- **Impact:** Theory of mind capabilities unavailable
- **Workaround:** Use `chat` endpoint for limited AI queries

### 4. ❌ `search_workspace`
- **Error:** `422 {"detail":"Input should be a valid dictionary or object..."}`
- **Issue:** Validation error on query parameter
- **Workaround:** Use `search_session_messages` or `search_peer_messages` instead

---

## Working Workflow

### Step 1: Initialize Session (First Time)
```
create_session
session_id: "session_unique_id"
config: {"purpose": "..."}  # optional

create_peer
peer_id: "user_peer"
config: {"role": "user"}  # optional

create_peer
peer_id: "assistant_peer"
config: {"role": "assistant"}  # optional

add_peers_to_session
session_id: "session_unique_id"
peer_ids: ["user_peer", "assistant_peer"]
```

### Step 2: Store Messages
```
add_messages_to_session
session_id: "session_unique_id"
messages: [
  {"peer_id": "user_peer", "content": "...", "metadata": {...}},
  {"peer_id": "assistant_peer", "content": "...", "metadata": {...}}
]
```

### Step 3: Query Understanding (Alternative to broken personalization)
```
chat
peer_id: "assistant_peer"
query: "What did the user say about X?"
session_id: "session_unique_id"  # optional, for session-scoped query
```

OR

```
get_working_representation
session_id: "session_unique_id"
peer_id: "assistant_peer"
target_peer_id: "user_peer"  # optional, to get what assistant knows about user
```

### Step 4: Retrieve History
```
get_session_messages
session_id: "session_unique_id"
filters: {...}  # optional

OR

get_session_context
session_id: "session_unique_id"
tokens: 5000
summary: true
```

---

## Key Discoveries

### 🎉 The `chat` Endpoint Works!
While `get_personalization_insights` is broken, the `chat` endpoint provides powerful AI-driven queries:
- Ask natural language questions about peer knowledge
- Get detailed, contextual responses
- Can scope to specific sessions
- Can query what one peer knows about another

**Example Response:**
```
Query: "What did the user say?"
Response: "Based on the available information, test_assistant_peer said 
they are running a comprehensive endpoint test. This is drawn from an 
explicit conclusion, meaning it directly reflects what was stated in 
their message from earlier today..."
```

### 📊 Working Representation Structure
The `get_working_representation` endpoint returns structured data:
```json
{
  "representation": {
    "explicit": [
      {
        "created_at": "2025-10-15T15:36:23Z",
        "message_ids": [[5,5]],
        "session_name": "full_test_session_001",
        "content": "test_user_peer is testing all Honcho endpoints"
      }
    ],
    "deductive": []
  }
}
```

---

## Recommendations

1. **Use the working workflow** documented above
2. **Use `chat` endpoint** instead of `get_personalization_insights` for AI queries
3. **Report bugs** to Honcho team:
   - `start_conversation` - 404 error
   - `add_turn` - 500 error
   - `get_personalization_insights` - 500 error
   - `search_workspace` - 422 validation error
4. **Monitor for server updates** that fix these issues

---

## Test Details

**Tested Endpoints:** 24/24  
**Working:** 20 (83%)  
**Broken:** 4 (17%)  

**Authentication:** ✅ Working (Bearer token + X-Honcho-User-Name)  
**JSON-RPC Protocol:** ✅ Working  
**Server Response:** ✅ Generally fast (except timeouts on broken endpoints)

