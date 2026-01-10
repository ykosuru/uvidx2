# Astra Code Search - VS Code Extension

Search legacy TAL/COBOL code and documentation using `@astra` in GitHub Copilot Chat.

## Installation

### Step 1: Install Dependencies

```bash
cd vscode-astra-extension
npm install
```

### Step 2: Compile

```bash
npm run compile
```

### Step 3: Install Extension

**Option A: Package and Install (Recommended)**
```bash
# Install packaging tool
npm install -g @vscode/vsce

# Package extension
vsce package

# This creates: astra-code-search-1.0.0.vsix
```

Then in VS Code:
1. Press `Ctrl+Shift+P`
2. Type "Install from VSIX"
3. Select `astra-code-search-1.0.0.vsix`

**Option B: Development Mode**
1. Open this folder in VS Code
2. Press `F5` to launch Extension Development Host
3. Test in the new VS Code window

### Step 4: Start API Server

Before using `@astra`, start the API server:

```bash
cd /path/to/unified-indexer
export INDEX_PATH=/path/to/your/index
python api_server.py
```

### Step 5: Configure Extension

In VS Code settings (`Ctrl+,`), set:

- `astra.apiUrl`: API server URL (default: `http://localhost:8080`)
- `astra.indexPath`: Path to your index (for starting server)
- `astra.defaultDomains`: Default domains to search (e.g., `["emts", "gmts"]`)

---

## Usage

In GitHub Copilot Chat, type:

### Search

```
@astra wire transfer validation
@astra OFAC sanctions screening
```

### Get Context for Code Generation

```
@astra /context implement OFAC screening for incoming wires
@astra /ctx add validation for MT103 messages
```

### Find Specific Procedure

```
@astra /procedure VALIDATE_WIRE_TRANSFER
@astra /proc OFAC_CHECK
```

### List Domains

```
@astra /domains
```

### Help

```
@astra /help
```

---

## Command Shortcuts

| Full Command | Shortcut |
|--------------|----------|
| `/search` | (default) |
| `/context` | `/c`, `/ctx` |
| `/procedure` | `/p`, `/proc` |
| `/domains` | `/d` |
| `/help` | `/h` |

---

## Workflow Example

1. **Ask Astra for context:**
   ```
   @astra /context implement OFAC screening for incoming wire transfers
   ```

2. **Review the code examples and documentation returned**

3. **Ask Copilot to generate code based on context:**
   ```
   Based on the examples above, create a new procedure SCREEN_INCOMING_WIRE 
   that validates the sender against the OFAC SDN list
   ```

4. **Search for related code if needed:**
   ```
   @astra find all procedures that call OFAC_CHECK
   ```

---

## Troubleshooting

### "API server is not running"

Start the server:
```bash
export INDEX_PATH=/path/to/idx
python api_server.py
```

### No results found

1. Check index has content: `python search_index.py --index ./idx --list-domains`
2. Try broader search terms
3. Check domain filter in settings

### Extension not showing in Copilot Chat

1. Ensure GitHub Copilot Chat is installed
2. Restart VS Code after installing extension
3. Check Output panel for errors (View → Output → Select "Astra Code Search")

---

## Development

```bash
# Watch mode for development
npm run watch

# Press F5 to launch Extension Development Host
```

---

## Architecture

```
┌─────────────────────────────────────┐
│         GitHub Copilot Chat         │
│                                     │
│   User: @astra wire transfer        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     Astra VS Code Extension         │
│                                     │
│   - Chat Participant Handler        │
│   - Command Parser                  │
│   - Results Formatter               │
└─────────────────┬───────────────────┘
                  │ HTTP
                  ▼
┌─────────────────────────────────────┐
│      api_server.py (FastAPI)        │
│                                     │
│   POST /search                      │
│   GET /domains                      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     Unified Indexer Pipeline        │
│                                     │
│   - Vector Search                   │
│   - BM25 Lexical                    │
│   - Domain Filtering                │
└─────────────────────────────────────┘
```
