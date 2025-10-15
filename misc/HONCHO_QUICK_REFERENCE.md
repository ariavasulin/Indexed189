# Honcho MCP Quick Reference

## 🎯 Status: 20/24 Endpoints Working (83%)

## ❌ Broken (4)
1. `start_conversation` → Use `create_session`
2. `add_turn` → Use `add_messages_to_session`  
3. `get_personalization_insights` → Use `chat`
4. `search_workspace` → Use `search_session_messages`

## ✅ Working Workflow

### 1. Initialize (First Time)
```
create_session
session_id: "session_001"

create_peer
peer_id: "user_peer"

create_peer
peer_id: "assistant_peer"

add_peers_to_session
session_id: "session_001"
peer_ids: ["user_peer", "assistant_peer"]
```

### 2. Store Messages
```
add_messages_to_session
session_id: "session_001"
messages: [
  {"peer_id": "user_peer", "content": "..."},
  {"peer_id": "assistant_peer", "content": "..."}
]
```

### 3. Query (AI Alternative to Personalization)
```
chat
peer_id: "assistant_peer"
query: "What did the user say about X?"
```

### 4. Retrieve
```
get_session_messages
session_id: "session_001"
```

## 🌟 Hidden Gem: `chat` Endpoint

The `chat` endpoint is **amazing** - it provides AI-powered natural language queries:

**Query:** "What did the user say about debugging?"  
**Response:** Detailed, contextual answer with citations and timestamps!

Use it as a replacement for the broken `get_personalization_insights`.

## 📊 All Working Endpoints (20)

**Sessions:** create_session, list_sessions, get/set_session_metadata, get_session_messages, get_session_context, get_session_peers

**Peers:** create_peer, list_peers, get/set_peer_metadata

**Session-Peer:** add/remove_peers_to/from_session

**Messages:** add_messages_to_session

**Search:** search_session_messages, search_peer_messages

**Workspace:** get/set_workspace_metadata

**AI:** chat ⭐, get_working_representation

## 📁 Full Documentation
- `HONCHO_ENDPOINT_TEST_RESULTS.md` - Complete test results
- `.rules/honcho.mdc` - Updated integration guide
- `.rules/honcho-workaround.md` - Technical analysis

