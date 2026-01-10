[README.md](https://github.com/user-attachments/files/24538229/README.md)
# Astra Code Search - VS Code Extension

Search legacy code using `@astra` in GitHub Copilot Chat.

## Install (No npm required!)

```bash
# 1. Install vsce globally (one time)
npm install -g @vscode/vsce

# 2. Package the extension
cd astra-extension-js
vsce package --allow-missing-repository

# 3. Install in VS Code
code --install-extension astra-code-search-1.0.0.vsix
```

Or manually:
1. Press `Ctrl+Shift+P` in VS Code
2. Type "Install from VSIX"
3. Select the `.vsix` file

## Before Using

Start the API server:
```bash
export INDEX_PATH=/path/to/idx
python api_server.py
```

## Usage

In Copilot Chat:

```
@astra wire transfer validation
@astra /context implement OFAC screening
@astra /procedure VALIDATE_WIRE
@astra /domains
@astra /help
```

## Configuration

In VS Code settings:
- `astra.apiUrl` - API server URL (default: http://localhost:8080)
- `astra.defaultDomains` - Default domains to filter
