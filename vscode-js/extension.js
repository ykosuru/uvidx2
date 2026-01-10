const vscode = require('vscode');
const http = require('http');
const https = require('https');

// Simple HTTP client using built-in Node modules (no axios needed)
function httpRequest(url, options = {}) {
    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(url);
        const client = parsedUrl.protocol === 'https:' ? https : http;
        
        const reqOptions = {
            hostname: parsedUrl.hostname,
            port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
            path: parsedUrl.pathname + parsedUrl.search,
            method: options.method || 'GET',
            headers: options.headers || {}
        };
        
        const req = client.request(reqOptions, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve({ status: res.statusCode, data: JSON.parse(data) });
                } catch {
                    resolve({ status: res.statusCode, data: data });
                }
            });
        });
        
        req.on('error', reject);
        req.setTimeout(10000, () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });
        
        if (options.body) {
            req.write(JSON.stringify(options.body));
        }
        req.end();
    });
}

// Get configuration
function getConfig() {
    const config = vscode.workspace.getConfiguration('astra');
    return {
        apiUrl: config.get('apiUrl') || 'http://localhost:8080',
        defaultDomains: config.get('defaultDomains') || []
    };
}

// API calls
async function searchCode(query, domains, topK = 10) {
    const config = getConfig();
    try {
        const response = await httpRequest(`${config.apiUrl}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: {
                query,
                top_k: topK,
                domains: domains && domains.length > 0 ? domains : undefined
            }
        });
        return response.data;
    } catch (error) {
        console.error('Search failed:', error);
        return null;
    }
}

async function listDomains() {
    const config = getConfig();
    try {
        const response = await httpRequest(`${config.apiUrl}/domains`);
        return response.data.domains || [];
    } catch (error) {
        console.error('Failed to list domains:', error);
        return [];
    }
}

async function checkHealth() {
    const config = getConfig();
    try {
        const response = await httpRequest(`${config.apiUrl}/health`);
        return response.data && response.data.status === 'healthy';
    } catch {
        return false;
    }
}

// Format results for chat
function formatResults(response) {
    if (!response || response.total_results === 0) {
        return `No results found for: "${response?.query || 'unknown'}"`;
    }
    
    let output = `## Found ${response.total_results} results for "${response.query}"\n\n`;
    
    for (const result of (response.results || []).slice(0, 10)) {
        output += `### ${result.rank}. ${result.procedure_name || 'Result'}\n`;
        output += `**File:** \`${result.file_path}\`\n`;
        
        if (result.line_start) {
            output += `**Lines:** ${result.line_start}-${result.line_end || result.line_start}\n`;
        }
        
        if (result.domain && result.domain !== 'default') {
            output += `**Domain:** ${result.domain}\n`;
        }
        
        output += `**Score:** ${result.score.toFixed(3)}\n\n`;
        output += '```\n' + (result.content_preview || '').substring(0, 800) + '\n```\n\n';
    }
    
    return output;
}

// Parse command
function parseCommand(query) {
    const trimmed = query.trim();
    
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
    
    return { command: 'search', args: trimmed };
}

// Chat handler
async function astraChatHandler(request, context, stream, token) {
    const config = getConfig();
    
    // Check API health
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        stream.markdown(`⚠️ **Astra API server is not running**\n\n`);
        stream.markdown(`Start the server:\n\`\`\`bash\nexport INDEX_PATH=/path/to/idx\npython api_server.py\n\`\`\`\n`);
        return { metadata: { command: 'error' } };
    }
    
    const { command, args } = parseCommand(request.prompt);
    
    switch (command) {
        case 'search':
        case 's': {
            if (!args) {
                stream.markdown('Provide a search query.\n\nExample: `@astra wire transfer validation`');
                return { metadata: { command: 'search' } };
            }
            
            stream.progress('Searching...');
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (results) {
                stream.markdown(formatResults(results));
            } else {
                stream.markdown('❌ Search failed.');
            }
            return { metadata: { command: 'search' } };
        }
        
        case 'context':
        case 'ctx':
        case 'c': {
            if (!args) {
                stream.markdown('Describe the task.\n\nExample: `@astra /context implement OFAC screening`');
                return { metadata: { command: 'context' } };
            }
            
            stream.progress('Gathering context...');
            
            const results = await searchCode(args, config.defaultDomains, 15);
            
            stream.markdown(`# Context for: ${args}\n\n`);
            
            if (results && results.results) {
                const docs = results.results.filter(r => r.source_type === 'document').slice(0, 3);
                const code = results.results.filter(r => r.source_type === 'code').slice(0, 5);
                
                if (docs.length > 0) {
                    stream.markdown('## Documentation\n\n');
                    for (const r of docs) {
                        stream.markdown(`### From: ${r.file_path}\n`);
                        stream.markdown((r.content_preview || '').substring(0, 1000) + '\n\n');
                    }
                }
                
                if (code.length > 0) {
                    stream.markdown('## Code Examples\n\n');
                    for (const r of code) {
                        stream.markdown(`### ${r.procedure_name || 'Code'} from \`${r.file_path}\`\n`);
                        stream.markdown('```\n' + (r.content_preview || '') + '\n```\n\n');
                    }
                }
            }
            
            return { metadata: { command: 'context' } };
        }
        
        case 'procedure':
        case 'proc':
        case 'p': {
            if (!args) {
                stream.markdown('Provide procedure name.\n\nExample: `@astra /procedure VALIDATE_WIRE`');
                return { metadata: { command: 'procedure' } };
            }
            
            stream.progress('Finding procedure...');
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (results && results.results && results.results.length > 0) {
                const matches = results.results.filter(r =>
                    r.procedure_name &&
                    r.procedure_name.toLowerCase().includes(args.toLowerCase())
                );
                
                const toShow = matches.length > 0 ? matches : results.results.slice(0, 3);
                
                stream.markdown(`## Procedure: ${args}\n\n`);
                for (const r of toShow.slice(0, 3)) {
                    stream.markdown(`### \`${r.procedure_name}\` in \`${r.file_path}\`\n`);
                    stream.markdown('```\n' + (r.content_preview || '') + '\n```\n\n');
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
                stream.markdown('| Domain | Chunks | % |\n|--------|--------|---|\n');
                for (const d of domains) {
                    stream.markdown(`| ${d.name} | ${d.chunk_count} | ${d.percentage.toFixed(1)}% |\n`);
                }
            } else {
                stream.markdown('No domains found.');
            }
            return { metadata: { command: 'domains' } };
        }
        
        case 'help':
        case 'h': {
            stream.markdown(`# Astra Help

| Command | Example |
|---------|---------|
| \`@astra <query>\` | \`@astra wire transfer\` |
| \`@astra /context <task>\` | \`@astra /context implement OFAC\` |
| \`@astra /procedure <name>\` | \`@astra /proc VALIDATE_WIRE\` |
| \`@astra /domains\` | \`@astra /domains\` |

**Shortcuts:** /c = context, /p = procedure, /d = domains
`);
            return { metadata: { command: 'help' } };
        }
        
        default: {
            // Default to search
            stream.progress('Searching...');
            const results = await searchCode(request.prompt, config.defaultDomains, 10);
            if (results) {
                stream.markdown(formatResults(results));
            }
            return { metadata: { command: 'search' } };
        }
    }
}

// Activation
function activate(context) {
    console.log('Astra Code Search activated');
    
    const astra = vscode.chat.createChatParticipant('astra.search', astraChatHandler);
    context.subscriptions.push(astra);
}

function deactivate() {}

module.exports = { activate, deactivate };
