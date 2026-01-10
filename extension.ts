import * as vscode from 'vscode';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

let mcpClient: Client;

export async function activate(context: vscode.ExtensionContext) {
    // Connect to your MCP server
    const transport = new StdioClientTransport({
        command: 'node',  // or 'python', depending on your server
        args: ['/path/to/your/mcp-server.js'],
        cwd: '/your/workspace'
    });

    mcpClient = new Client({ name: 'devx-vscode', version: '1.0.0' });
    await mcpClient.connect(transport);

    // Register chat participant
    const handler: vscode.ChatRequestHandler = async (request, context, stream, token) => {
        const query = request.prompt;

        try {
            // Call your MCP search tool
            const result = await mcpClient.callTool({
                name: 'search',  // your tool name
                arguments: { query }
            });

            const searchResults = JSON.parse(result.content[0].text);

            if (searchResults.length === 0) {
                stream.markdown('No results found.');
                return {};
            }

            // Option A: Just show results
            stream.markdown(`**Found ${searchResults.length} results:**\n\n`);
            for (const r of searchResults) {
                stream.markdown(`- **${r.filename}**: ${r.snippet}\n`);
            }

            // Option B: Pass to Copilot LLM to synthesize an answer
            const contextText = searchResults.map(r => r.content).join('\n\n---\n\n');
            const models = await vscode.lm.selectChatModels({ family: 'gpt-4o' });
            
            if (models.length > 0) {
                const messages = [
                    vscode.LanguageModelChatMessage.User(
                        `Based on this context:\n\n${contextText}\n\nAnswer: ${query}`
                    )
                ];
                
                const response = await models[0].sendRequest(messages, {}, token);
                stream.markdown('\n\n**Answer:**\n');
                for await (const chunk of response.text) {
                    stream.markdown(chunk);
                }
            }

        } catch (err) {
            stream.markdown(`Error: ${err.message}`);
        }

        return {};
    };

    const participant = vscode.chat.createChatParticipant('devx.assistant', handler);
    context.subscriptions.push(participant);
}

export async function deactivate() {
    await mcpClient?.close();
}
