#!/bin/bash
# Astra VS Code Extension - Build and Install Script

set -e

echo "========================================"
echo "  Astra Code Search Extension Installer"
echo "========================================"
echo

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "   Install from: https://nodejs.org/"
    exit 1
fi

echo "✓ Node.js found: $(node --version)"

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed"
    exit 1
fi

echo "✓ npm found: $(npm --version)"

# Install dependencies
echo
echo "Installing dependencies..."
npm install

# Compile TypeScript
echo
echo "Compiling TypeScript..."
npm run compile

# Check if vsce is installed
if ! command -v vsce &> /dev/null; then
    echo
    echo "Installing vsce (VS Code Extension packaging tool)..."
    npm install -g @vscode/vsce
fi

# Package extension
echo
echo "Packaging extension..."
vsce package --allow-missing-repository

# Find the vsix file
VSIX_FILE=$(ls -t *.vsix 2>/dev/null | head -1)

if [ -z "$VSIX_FILE" ]; then
    echo "❌ Failed to create .vsix package"
    exit 1
fi

echo
echo "✓ Created: $VSIX_FILE"

# Try to install
echo
echo "Installing extension in VS Code..."
if command -v code &> /dev/null; then
    code --install-extension "$VSIX_FILE"
    echo
    echo "✓ Extension installed!"
    echo
    echo "Next steps:"
    echo "  1. Restart VS Code"
    echo "  2. Start the API server: python api_server.py"
    echo "  3. Open Copilot Chat and type: @astra /help"
else
    echo "⚠️  VS Code CLI not found"
    echo
    echo "To install manually:"
    echo "  1. Open VS Code"
    echo "  2. Press Ctrl+Shift+P"
    echo "  3. Type: Install from VSIX"
    echo "  4. Select: $VSIX_FILE"
fi

echo
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
