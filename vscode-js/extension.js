const vscode = require('vscode');
const http = require('http');
const https = require('https');

// Simple HTTP client using built-in Node modules
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

// Build context from search results
function buildContext(results) {
    if (!results || !results.results || results.results.length === 0) {
        return null;
    }
    
    let context = '';
    for (const result of results.results.slice(0, 8)) {
        const source = result.procedure_name 
            ? `${result.procedure_name} (${result.file_path})`
            : result.file_path;
        
        context += `\n--- Source: ${source} ---\n`;
        if (result.line_start) {
            context += `Lines: ${result.line_start}-${result.line_end || result.line_start}\n`;
        }
        context += `${result.content_preview || ''}\n`;
    }
    return context;
}

// System prompts for different modes
const SYSTEM_PROMPTS = {
    search: `You are Astra, an expert assistant for legacy payment systems code (TAL, COBOL) and ISO 20022 documentation.

You help developers understand, modernize, and work with legacy code. When answering:
- Reference specific code snippets and file locations from the provided context
- Explain legacy patterns and their modern equivalents
- Be precise about procedure names, variables, and data flows
- If the context doesn't contain enough information, say so clearly

Keep responses focused and actionable.`,

    context: `You are Astra, an expert assistant for legacy payment systems.

Your task is to provide comprehensive context for implementing or understanding a feature. Using the provided search results:
1. Summarize relevant documentation
2. Identify key procedures and their purposes
3. Note important data structures and flows
4. Highlight any validation or business rules
5. Suggest which code to reference for implementation

Be thorough but organized.`,

    explain: `You are Astra, an expert at explaining legacy code.

Analyze the provided code and explain:
1. What the code does (high-level purpose)
2. Key logic and control flow
3. Important variables and data structures
4. Any business rules or validations
5. Potential issues or modernization opportunities

Use clear, accessible language.`,

    modernize: `You are Astra, an expert at modernizing legacy systems.

Given the legacy code context, provide:
1. Modern equivalent patterns (in Python, Java, or TypeScript as appropriate)
2. Key architectural changes needed
3. Data structure mappings
4. Testing considerations
5. Migration risks and mitigations

Focus on practical, implementable guidance.`
};

// Get language model
async function getLanguageModel() {
    try {
        const models = await vscode.lm.selectChatModels({
            vendor: 'copilot',
            family: 'gpt-4o'
        });
        
        if (models.length > 0) {
            return models[0];
        }
        
        // Fallback to any available model
        const allModels = await vscode.lm.selectChatModels({ vendor: 'copilot' });
        return allModels.length > 0 ? allModels[0] : null;
    } catch (error) {
        console.error('Failed to get language model:', error);
        return null;
    }
}

// Send to LLM and stream response
async function queryLLM(model, systemPrompt, userPrompt, context, stream, token) {
    const messages = [
        vscode.LanguageModelChatMessage.User(`${systemPrompt}\n\n---\n\nCONTEXT FROM CODEBASE:\n${context}`),
        vscode.LanguageModelChatMessage.User(userPrompt)
    ];
    
    try {
        const response = await model.sendRequest(messages, {}, token);
        
        for await (const chunk of response.text) {
            if (token.isCancellationRequested) break;
            stream.markdown(chunk);
        }
        return true;
    } catch (error) {
        if (error.code === 'ContentFiltered') {
            stream.markdown('\n\n⚠️ Response was filtered by content policy.');
        } else {
            console.error('LLM error:', error);
            stream.markdown(`\n\n❌ LLM error: ${error.message}`);
        }
        return false;
    }
}

// Format results for display (non-LLM fallback)
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
    
    // Get LLM
    const model = await getLanguageModel();
    const hasLLM = model !== null;
    
    const { command, args } = parseCommand(request.prompt);
    
    switch (command) {
        case 'search':
        case 's': {
            if (!args) {
                stream.markdown('Provide a search query.\n\nExample: `@astra wire transfer validation`');
                return { metadata: { command: 'search' } };
            }
            
            stream.progress('Searching codebase...');
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (!results || results.total_results === 0) {
                stream.markdown(`No results found for: "${args}"`);
                return { metadata: { command: 'search' } };
            }
            
            const codeContext = buildContext(results);
            
            if (hasLLM && codeContext) {
                stream.progress('Analyzing results...');
                await queryLLM(
                    model,
                    SYSTEM_PROMPTS.search,
                    `Based on the codebase context above, answer this question: ${args}`,
                    codeContext,
                    stream,
                    token
                );
                
                // Add sources
                stream.markdown('\n\n---\n**Sources:**\n');
                for (const r of results.results.slice(0, 5)) {
                    const name = r.procedure_name || r.file_path.split('/').pop();
                    stream.markdown(`- \`${name}\` in \`${r.file_path}\`\n`);
                }
            } else {
                stream.markdown(formatResults(results));
            }
            
            return { metadata: { command: 'search', resultsCount: results.total_results } };
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
            
            if (!results || results.total_results === 0) {
                stream.markdown(`No relevant context found for: "${args}"`);
                return { metadata: { command: 'context' } };
            }
            
            const codeContext = buildContext(results);
            
            if (hasLLM && codeContext) {
                stream.progress('Building comprehensive context...');
                await queryLLM(
                    model,
                    SYSTEM_PROMPTS.context,
                    `Provide comprehensive context for this task: ${args}`,
                    codeContext,
                    stream,
                    token
                );
                
                stream.markdown('\n\n---\n**Referenced files:**\n');
                const files = [...new Set(results.results.map(r => r.file_path))].slice(0, 8);
                for (const f of files) {
                    stream.markdown(`- \`${f}\`\n`);
                }
            } else {
                // Fallback to raw results
                stream.markdown(`# Context for: ${args}\n\n`);
                stream.markdown(formatResults(results));
            }
            
            return { metadata: { command: 'context' } };
        }
        
        case 'explain':
        case 'e': {
            if (!args) {
                stream.markdown('Specify what to explain.\n\nExample: `@astra /explain VALIDATE_WIRE procedure`');
                return { metadata: { command: 'explain' } };
            }
            
            stream.progress('Finding code...');
            const results = await searchCode(args, config.defaultDomains, 8);
            
            if (!results || results.total_results === 0) {
                stream.markdown(`No code found for: "${args}"`);
                return { metadata: { command: 'explain' } };
            }
            
            const codeContext = buildContext(results);
            
            if (hasLLM && codeContext) {
                stream.progress('Analyzing code...');
                await queryLLM(
                    model,
                    SYSTEM_PROMPTS.explain,
                    `Explain this code/concept in detail: ${args}`,
                    codeContext,
                    stream,
                    token
                );
            } else {
                stream.markdown(formatResults(results));
            }
            
            return { metadata: { command: 'explain' } };
        }
        
        case 'modernize':
        case 'mod':
        case 'm': {
            if (!args) {
                stream.markdown('Specify what to modernize.\n\nExample: `@astra /modernize wire transfer validation`');
                return { metadata: { command: 'modernize' } };
            }
            
            stream.progress('Analyzing legacy code...');
            const results = await searchCode(args, config.defaultDomains, 10);
            
            if (!results || results.total_results === 0) {
                stream.markdown(`No code found for: "${args}"`);
                return { metadata: { command: 'modernize' } };
            }
            
            const codeContext = buildContext(results);
            
            if (hasLLM && codeContext) {
                stream.progress('Generating modernization guidance...');
                await queryLLM(
                    model,
                    SYSTEM_PROMPTS.modernize,
                    `Provide modernization guidance for: ${args}`,
                    codeContext,
                    stream,
                    token
                );
            } else {
                stream.markdown('⚠️ LLM required for modernization analysis.\n\n');
                stream.markdown(formatResults(results));
            }
            
            return { metadata: { command: 'modernize' } };
        }
        
        case 'raw':
        case 'r': {
            // Raw search without LLM processing
            if (!args) {
                stream.markdown('Provide a search query.\n\nExample: `@astra /raw wire transfer`');
                return { metadata: { command: 'raw' } };
            }
            
            stream.progress('Searching...');
            const results = await searchCode(args, config.defaultDomains, 10);
            stream.markdown(formatResults(results));
            return { metadata: { command: 'raw' } };
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
                const codeContext = buildContext({ results: toShow });
                
                if (hasLLM && codeContext) {
                    stream.progress('Analyzing procedure...');
                    await queryLLM(
                        model,
                        SYSTEM_PROMPTS.explain,
                        `Explain this procedure in detail: ${args}`,
                        codeContext,
                        stream,
                        token
                    );
                    
                    stream.markdown('\n\n---\n**Source code:**\n');
                    for (const r of toShow.slice(0, 3)) {
                        stream.markdown(`\n### \`${r.procedure_name}\` in \`${r.file_path}\`\n`);
                        stream.markdown('```\n' + (r.content_preview || '').substring(0, 1000) + '\n```\n');
                    }
                } else {
                    stream.markdown(`## Procedure: ${args}\n\n`);
                    for (const r of toShow.slice(0, 3)) {
                        stream.markdown(`### \`${r.procedure_name}\` in \`${r.file_path}\`\n`);
                        stream.markdown('```\n' + (r.content_preview || '') + '\n```\n\n');
                    }
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

## Commands with LLM Analysis

| Command | Description | Example |
|---------|-------------|---------|
| \`@astra <query>\` | Search + AI analysis | \`@astra wire transfer\` |
| \`@astra /context <task>\` | Comprehensive context | \`@astra /c implement OFAC\` |
| \`@astra /explain <code>\` | Explain code | \`@astra /e VALIDATE_WIRE\` |
| \`@astra /modernize <code>\` | Modernization guide | \`@astra /m payment validation\` |
| \`@astra /procedure <name>\` | Find & explain proc | \`@astra /p VALIDATE_WIRE\` |

## Utility Commands

| Command | Description |
|---------|-------------|
| \`@astra /raw <query>\` | Search without LLM |
| \`@astra /domains\` | List indexed domains |
| \`@astra /help\` | Show this help |

**Shortcuts:** /c=context, /e=explain, /m=modernize, /p=procedure, /r=raw, /d=domains

**LLM Status:** ${hasLLM ? '✅ Connected' : '❌ Not available'}
`);
            return { metadata: { command: 'help' } };
        }
        
        default: {
            // Default to search with LLM
            stream.progress('Searching...');
            const results = await searchCode(request.prompt, config.defaultDomains, 10);
            
            if (results && results.total_results > 0) {
                const codeContext = buildContext(results);
                
                if (hasLLM && codeContext) {
                    stream.progress('Analyzing...');
                    await queryLLM(
                        model,
                        SYSTEM_PROMPTS.search,
                        `Answer this question based on the codebase: ${request.prompt}`,
                        codeContext,
                        stream,
                        token
                    );
                    
                    stream.markdown('\n\n---\n**Sources:**\n');
                    for (const r of results.results.slice(0, 5)) {
                        const name = r.procedure_name || r.file_path.split('/').pop();
                        stream.markdown(`- \`${name}\` in \`${r.file_path}\`\n`);
                    }
                } else {
                    stream.markdown(formatResults(results));
                }
            } else {
                stream.markdown(`No results found for: "${request.prompt}"`);
            }
            return { metadata: { command: 'search' } };
        }
    }
}

// Activation
function activate(context) {
    console.log('Astra Code Search activated');
    
    const astra = vscode.chat.createChatParticipant('astra.search', astraChatHandler);
    astra.iconPath = vscode.Uri.joinPath(context.extensionUri, 'icon.png');
    context.subscriptions.push(astra);
}

function deactivate() {}

module.exports = { activate, deactivate };
