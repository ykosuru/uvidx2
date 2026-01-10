import * as vscode from 'vscode';
import axios from 'axios';

// Types for API responses
interface SearchResult {
    rank: number;
    score: number;
    file_path: string;
    domain: string;
    source_type: string;
    procedure_name?: string;
    line_start?: number;
    line_end?: number;
    content_preview: string;
    concepts: string[];
}

interface SearchResponse {
    query: string;
    total_results: number;
    search_time_ms: number;
    results: SearchResult[];
}

interface DomainInfo {
    name: string;
    chunk_count: number;
    percentage: number;
}

// Get configuration
function getConfig() {
    const config = vscode.workspace.getConfiguration('astra');
    return {
        apiUrl: config.get<string>('apiUrl') || 'http://localhost:8080',
        indexPath: config.get<string>('indexPath') || '',
        defaultDomains: config.get<string[]>('defaultDomains') || []
    };
}

// API client
async function searchCode(query: string, domains?: string[], topK: number = 10): Promise<SearchResponse | null> {
    const config = getConfig();
    
    try {
        const response = await axios.post(`${config.apiUrl}/search`, {
            query,
            top_k: topK,
            domains: domains && domains.length > 0 ? domains : undefined
        });
        return response.data;
    } catch (error) {
        console.error('Search failed:', error);
        return null;
    }
}

async function listDomains(): Promise<DomainInfo[]> {
    const config = getConfig();
    
    try {
        const response = await axios.get(`${config.apiUrl}/domains`);
        return response.data.domains || [];
    } catch (error) {
        console.error('Failed to list domains:', error);
        return [];
    }
}

async function checkHealth(): Promise<boolean> {
    const config = getConfig();
    
    try {
        const response = await axios.get(`${config.apiUrl}/health`, { timeout: 2000 });
        return response.data.status === 'healthy';
    } catch {
        return false;
    }
}

// Format search results for chat
function formatResults(response: SearchResponse): string {
    if (response.total_results === 0) {
        return `No results found for: "${response.query}"`;
    }
    
    let output = `## Found ${response.total_results} results for "${response.query}"\n\n`;
    
    for (const result of response.results.slice(0, 10)) {
        output += `### ${result.rank}. ${result.procedure_name || 'Result'}\n`;
        output += `**File:** \`${result.file_path}\`\n`;
        
        if (result.line_start) {
            output += `**Lines:** ${result.line_start}-${result.line_end || result.line_start}\n`;
        }
        
        if (result.domain && result.domain !== 'default') {
            output += `**Domain:** ${result.domain}\n`;
        }
        
        output += `**Score:** ${result.score.toFixed(3)}\n\n`;
        output += '```\n' + result.content_preview.substring(0, 800) + '\n```\n\n';
    }
    
    return output;
}

// Parse command from user query
function parseCommand(query: string): { command: string; args: string } {
    const trimmed = query.trim();
    
    // Check for /command syntax
    if (trimmed.startsWith('/')) {
        const spaceIndex = trimmed.indexOf(' ');
        if (spaceIndex > 0) {
            return {
                command: trimmed.substring(1, spaceIndex).toLowerCase(),
                args: trimmed.substring(spaceIndex + 1).trim()
            };
        }
        return { command: trimmed.substring(1).toLowerCase(), args: '' };
    }
    
    // Default to search
    return { command: 'search', args: trimmed };
}

// Chat participant handler
const astraChatHandler: vscode.ChatRequestHandler = async (
    request: vscode.ChatRequest,
    context: vscode.ChatContext,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken
): Promise<vscode.ChatResult> => {
    
    const config = getConfig();
    
    // Check if API is available
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        stream.markdown(`⚠️ **Astra API server is not running**\n\n`);
        stream.markdown(`Start the server with:\n\`\`\`bash\ncd /path/to/unified-indexer\nexport INDEX_PATH=/path/to/idx\npython api_server.py\n\`\`\`\n\n`);
        stream.markdown(`Or configure the API URL in settings: \`astra.apiUrl\``);
        return { metadata: { command: 'error' } };
    }
    
    const { command, args } = parseCommand(request.prompt);
    
    // Handle commands
    switch (command) {
        case 'search':
        case 's': {
            if (!args) {
                stream.markdown('Please provide a search query.\n\nExample: `@astra wire transfer validation`');
                return { metadata: { command: 'search' } };
            }
            
            stream.progress('Searching...');
            
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (results) {
                stream.markdown(formatResults(results));
                
                // Add suggestion to use context
                if (results.total_results > 0) {
                    stream.markdown('\n💡 *Use `/context <task>` to get comprehensive context for code generation.*');
                }
            } else {
                stream.markdown('❌ Search failed. Check that the API server is running.');
            }
            
            return { metadata: { command: 'search' } };
        }
        
        case 'context':
        case 'ctx':
        case 'c': {
            if (!args) {
                stream.markdown('Please describe the task.\n\nExample: `@astra /context implement OFAC screening for incoming wires`');
                return { metadata: { command: 'context' } };
            }
            
            stream.progress('Gathering context...');
            
            // Search for documentation
            const docs = await searchCode(args, config.defaultDomains, 5);
            
            // Search for code
            const code = await searchCode(args, config.defaultDomains, 10);
            
            stream.markdown(`# Context for: ${args}\n\n`);
            
            if (docs && docs.total_results > 0) {
                stream.markdown('## Relevant Documentation\n\n');
                for (const result of docs.results.filter(r => r.source_type === 'document').slice(0, 3)) {
                    stream.markdown(`### From: ${result.file_path}\n`);
                    stream.markdown(result.content_preview.substring(0, 1000) + '\n\n');
                }
            }
            
            if (code && code.total_results > 0) {
                stream.markdown('## Code Examples\n\n');
                for (const result of code.results.filter(r => r.source_type === 'code').slice(0, 5)) {
                    stream.markdown(`### ${result.procedure_name || 'Code'} from \`${result.file_path}\`\n`);
                    if (result.line_start) {
                        stream.markdown(`Lines ${result.line_start}-${result.line_end || result.line_start}\n`);
                    }
                    stream.markdown('```\n' + result.content_preview + '\n```\n\n');
                }
            }
            
            stream.markdown('\n---\n*Use this context to generate code following existing patterns.*');
            
            return { metadata: { command: 'context' } };
        }
        
        case 'procedure':
        case 'proc':
        case 'p': {
            if (!args) {
                stream.markdown('Please provide a procedure name.\n\nExample: `@astra /procedure VALIDATE_WIRE_TRANSFER`');
                return { metadata: { command: 'procedure' } };
            }
            
            stream.progress('Finding procedure...');
            
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (results && results.total_results > 0) {
                // Filter for matching procedure names
                const matches = results.results.filter(r => 
                    r.procedure_name && 
                    r.procedure_name.toLowerCase().includes(args.toLowerCase())
                );
                
                const toShow = matches.length > 0 ? matches : results.results.slice(0, 3);
                
                stream.markdown(`## Procedure: ${args}\n\n`);
                
                for (const result of toShow.slice(0, 3)) {
                    stream.markdown(`### \`${result.procedure_name}\` in \`${result.file_path}\`\n`);
                    if (result.line_start) {
                        stream.markdown(`Lines ${result.line_start}-${result.line_end || result.line_start}\n`);
                    }
                    stream.markdown('```\n' + result.content_preview + '\n```\n\n');
                }
            } else {
                stream.markdown(`Procedure not found: ${args}`);
            }
            
            return { metadata: { command: 'procedure' } };
        }
        
        case 'domains':
        case 'd': {
            stream.progress('Loading domains...');
            
            const domains = await listDomains();
            
            if (domains.length > 0) {
                stream.markdown('## Available Domains\n\n');
                stream.markdown('| Domain | Chunks | % |\n');
                stream.markdown('|--------|--------|---|\n');
                
                for (const domain of domains) {
                    stream.markdown(`| ${domain.name} | ${domain.chunk_count} | ${domain.percentage.toFixed(1)}% |\n`);
                }
                
                stream.markdown('\n💡 *Set default domains in settings: `astra.defaultDomains`*');
            } else {
                stream.markdown('No domains found in index.');
            }
            
            return { metadata: { command: 'domains' } };
        }
        
        case 'help':
        case 'h': {
            stream.markdown(`# Astra Code Search Help

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| \`@astra <query>\` | Search code and docs | \`@astra wire transfer validation\` |
| \`@astra /context <task>\` | Get context for code generation | \`@astra /context implement OFAC screening\` |
| \`@astra /procedure <name>\` | Find specific procedure | \`@astra /procedure VALIDATE_WIRE\` |
| \`@astra /domains\` | List available domains | \`@astra /domains\` |

## Shortcuts

- \`/s\` = /search
- \`/c\` or \`/ctx\` = /context  
- \`/p\` or \`/proc\` = /procedure
- \`/d\` = /domains

## Configuration

Set in VS Code settings:
- \`astra.apiUrl\` - API server URL (default: http://localhost:8080)
- \`astra.defaultDomains\` - Default domains to search
`);
            return { metadata: { command: 'help' } };
        }
        
        default: {
            // Treat as search query
            stream.progress('Searching...');
            
            const results = await searchCode(request.prompt, config.defaultDomains, 10);
            
            if (results) {
                stream.markdown(formatResults(results));
            } else {
                stream.markdown('❌ Search failed.');
            }
            
            return { metadata: { command: 'search' } };
        }
    }
};

// Extension activation
export function activate(context: vscode.ExtensionContext) {
    console.log('Astra Code Search extension activated');
    
    // Register chat participant
    const astra = vscode.chat.createChatParticipant('astra.search', astraChatHandler);
    astra.iconPath = vscode.Uri.joinPath(context.extensionUri, 'icon.png');
    
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('astra.startServer', async () => {
            const terminal = vscode.window.createTerminal('Astra Server');
            const config = getConfig();
            
            terminal.show();
            terminal.sendText(`export INDEX_PATH="${config.indexPath}"`);
            terminal.sendText('python api_server.py');
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('astra.setIndex', async () => {
            const uri = await vscode.window.showOpenDialog({
                canSelectFiles: false,
                canSelectFolders: true,
                canSelectMany: false,
                title: 'Select Index Directory'
            });
            
            if (uri && uri[0]) {
                const config = vscode.workspace.getConfiguration('astra');
                await config.update('indexPath', uri[0].fsPath, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(`Index path set to: ${uri[0].fsPath}`);
            }
        })
    );
    
    context.subscriptions.push(astra);
}

export function deactivate() {}
