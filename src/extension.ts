import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

// This method is called when your extension is activated
export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "llm-napkin" is now active!');

    // Register the command
    const disposable = vscode.commands.registerCommand('huggingfaceConfigLoader.loadConfig', async () => {
        try {
            // Open file picker
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                openLabel: 'Select Hugging Face Config',
                filters: { 'JSON Files': ['json'] }
            });

            if (!fileUri || fileUri.length === 0) {
                vscode.window.showErrorMessage('No file selected.');
                return;
            }

            // Read and parse the JSON file
            const filePath = fileUri[0].fsPath;
            const fileContent = fs.readFileSync(filePath, 'utf-8');
            const config = JSON.parse(fileContent);

            // Display the parsed configuration
            vscode.window.showInformationMessage('Hugging Face Config Loaded Successfully!');
            console.log('Hugging Face Config:', config);
        } catch (error) {
            vscode.window.showErrorMessage('Failed to load Hugging Face Config.');
            console.error(error);
        }
    });

    context.subscriptions.push(disposable);

    // Register the webview view provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('huggingfaceInputs', new HuggingFaceInputViewProvider(context))
    );
}

class HuggingFaceInputViewProvider implements vscode.WebviewViewProvider {
    constructor(private readonly context: vscode.ExtensionContext) {}

    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.options = {
            enableScripts: true
        };

        webviewView.webview.html = this.getHtmlForWebview();

        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'submit':
                    try {
                        const data = message.data;
                        const modelPath = data.modelPath;
                        
                        // Extract model config URL from model path
                        const configUrl = this.getConfigUrl(modelPath);
                        
                        // Fetch config from Hugging Face
                        const config = await this.fetchModelConfig(configUrl, data.apiKey);
                        
                        // Calculate memory usage
                        const memoryUsage = this.calculateMemoryUsage(
                            config, 
                            parseInt(data.inputLength), 
                            parseInt(data.outputLength),
                            parseInt(data.batchSize),
                            data.quant
                        );
                        
                        // Send results back to the webview
                        webviewView.webview.postMessage({ 
                            command: 'results',
                            config,
                            memoryUsage
                        });
                        
                        vscode.window.showInformationMessage(`Configuration loaded for ${modelPath}`);
                    } catch (error) {
                        console.error(error);
                        vscode.window.showErrorMessage(`Error: ${error instanceof Error ? error.message : String(error)}`);
                        webviewView.webview.postMessage({ 
                            command: 'error',
                            message: error instanceof Error ? error.message : String(error)
                        });
                    }
                    return;
            }
        });
    }

    private getConfigUrl(modelPath: string): string {
        // Clean up the model path to ensure it's in the correct format
        const cleanPath = modelPath.trim()
            .replace('https://huggingface.co/', '')
            .replace(/\/blob\/main\/?.*/, '');
            
        return `https://huggingface.co/${cleanPath}/raw/main/config.json`;
    }

    private fetchModelConfig(url: string, apiKey?: string): Promise<any> {
        return new Promise((resolve, reject) => {
            const options: https.RequestOptions = {
                headers: {}
            };
            
            // Add Authorization header if API key is provided
            if (apiKey && apiKey.trim() !== '') {
                options.headers = {
                    'Authorization': `Bearer ${apiKey}`
                };
            }

            https.get(url, options, (res) => {
                if (res.statusCode !== 200) {
                    if (res.statusCode === 401) {
                        reject(new Error('Authentication required. Please provide a valid Hugging Face API key.'));
                    } else if (res.statusCode === 404) {
                        reject(new Error('Model configuration not found. Please check the model path.'));
                    } else {
                        reject(new Error(`Failed to fetch model config. Status code: ${res.statusCode}`));
                    }
                    return;
                }

                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const config = JSON.parse(data);
                        resolve(config);
                    } catch (e) {
                        reject(new Error('Failed to parse model config'));
                    }
                });
            }).on('error', reject);
        });
    }

    private calculateMemoryUsage(
        config: any, 
        inputLength: number,
        outputLength: number,
        batchSize: number,
        quantization: string
    ): any {
        // Extract relevant parameters from config
        const D = config.hidden_size || config.d_model || 0; // Hidden dimension
        const L = config.num_hidden_layers || config.n_layer || 0; // Number of layers
        const V = config.vocab_size || 0; // Vocabulary size
        const h = config.num_attention_heads || config.n_head || 0; // Number of attention heads
        const k = config.num_key_value_heads || h; // Number of KV heads (for GQA)
        const F = config.intermediate_size || config.n_inner || 4 * D; // Feed-forward dimension
        const T = inputLength + outputLength; // Sequence length
        const B = batchSize; // Batch size

        // Compute r = k/h (KV-to-Q-head ratio) for GQA
        const r = k / h;
        
        // Calculate bytes per parameter based on quantization
        let b = 2; // default for float16/FP16/bfloat16
        if (quantization === 'int8' || quantization === 'fp8') {
            b = 1;
        } else if (quantization === 'int4' || quantization === 'fp4') {
            b = 0.5;
        } else if (quantization === 'fp32') {
            b = 4;
        }

        // Calculate total parameters using the generalized formula:
        // P = V×D + L×[(2+r)D² + 3DF]
        const embeddingParams = V * D; // Token embeddings
        
        // For attention: (2+r)D² instead of 4D² for GQA
        const attentionParams = (2 + r) * Math.pow(D, 2);
        
        // For FFN: 3DF instead of 2DF for GLU variants
        const ffnParams = 3 * D * F;
        
        // Total per layer
        const paramsPerLayer = attentionParams + ffnParams;
        
        // Total parameters
        const totalParams = embeddingParams + (L * paramsPerLayer);
        
        // Calculate memory requirements
        
        // 1. Weights memory: M_weights = P × b (bytes)
        const weightsSizeBytes = totalParams * b;
        
        // 2. Activation memory (for inference): M_act = B × T × D × b (bytes)
        const inferenceActivationSizeBytes = B * T * D * b;
        
        // 3. Full activation memory (for training): M_act = B * L * D * (T + (2 * D / h)) * b (bytes)
        const trainingActivationSizeBytes = B * L * D * (T + (2 * D / h)) * b;
        
        // Convert to MB and GB
        const weightsSizeMB = weightsSizeBytes / (1024 * 1024);
        const inferenceActivationSizeMB = inferenceActivationSizeBytes / (1024 * 1024);
        const trainingActivationSizeMB = trainingActivationSizeBytes / (1024 * 1024);
        
        const inferenceMemoryGB = (weightsSizeMB + inferenceActivationSizeMB) / 1024;
        const trainingMemoryGB = (weightsSizeMB + trainingActivationSizeMB) / 1024;
        
        return {
            modelSizeMB: weightsSizeMB.toFixed(2),
            inferenceActivationSizeMB: inferenceActivationSizeMB.toFixed(2),
            trainingActivationSizeMB: trainingActivationSizeMB.toFixed(2),
            inferenceMemoryGB: inferenceMemoryGB.toFixed(2),
            trainingMemoryGB: trainingMemoryGB.toFixed(2),
            details: {
                parameters: totalParams.toLocaleString(),
                parametersBillions: (totalParams / 1000000000).toFixed(2) + "B",
                hiddenSize: D,
                numLayers: L, 
                numHeads: h,
                numKVHeads: k,
                kvRatio: r.toFixed(2),
                ffnSize: F,
                vocabSize: V,
                bytesPerParam: b,
                quantization,
                batchSize: B,
                sequenceLength: T
            }
        };
    }

    private getHtmlForWebview(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM-Napkin Config</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            padding: 10px;
        }
        label {
            display: block;
            margin: 8px 0;
        }
        input, select {
            width: 100%;
            margin-bottom: 10px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 5px;
        }
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 12px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .results {
            margin-top: 20px;
            padding: 10px;
            background: var(--vscode-panel-background);
            border: 1px solid var(--vscode-panel-border);
            display: none;
        }
        .memory-box {
            margin-top: 15px;
            padding: 8px;
            background: var(--vscode-panel-background);
            border-left: 3px solid var(--vscode-activityBarBadge-background);
        }
    </style>
</head>
<body>
    <h2>LLM-Napkin Model Configuration</h2>
    <form id="configForm">
        <label>HF API Key: <input type="password" id="apiKey" placeholder="Optional, for private models"></label>
        
        <label>Model Path: <input type="text" id="modelPath" required 
            placeholder="e.g., Qwen/Qwen3-4B or full URL"></label>
        
        <label>Quantization: 
            <select id="quant">
                <option value="float16">float16</option>
                <option value="int8">int8/fp8</option>
                <option value="int4">int4/fp4</option>
            </select>
        </label>
        
        <label>Input Token Length: <input type="number" id="inputLength" value="128"></label>
        
        <label>Output Token Length: <input type="number" id="outputLength" value="128"></label>
        
        <label>Batch Size: <input type="number" id="batchSize" value="1"></label>
        
        <button type="button" onclick="submitForm()">Calculate Memory Usage</button>
    </form>
    
    <div id="results" class="results">
        <h3>Memory Usage Estimation</h3>
        <div id="memory-usage"></div>
        <div id="model-info"></div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        
        function submitForm() {
            const data = {
                apiKey: document.getElementById('apiKey').value,
                modelPath: document.getElementById('modelPath').value,
                quant: document.getElementById('quant').value,
                inputLength: document.getElementById('inputLength').value,
                outputLength: document.getElementById('outputLength').value,
                batchSize: document.getElementById('batchSize').value
            };
            
            document.getElementById('results').style.display = 'none';
            document.getElementById('memory-usage').innerHTML = '<p>Loading...</p>';
            
            vscode.postMessage({ command: 'submit', data });
        }
        
        window.addEventListener('message', event => {
            const message = event.data;
            
            switch (message.command) {
                case 'results':
                    displayResults(message.config, message.memoryUsage);
                    break;
                case 'error':
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('memory-usage').innerHTML = 
                        \`<p style="color:var(--vscode-errorForeground)">Error: \${message.message}</p>\`;
                    break;
            }
        });
        
        function displayResults(config, memoryUsage) {
            document.getElementById('results').style.display = 'block';
            
            // Display memory usage
            let memoryHtml = \`
                <div class="memory-box">
                    <strong>Total Memory Required:</strong> \${memoryUsage.trainingMemoryGB} GB<br>
                    <strong>Model Size:</strong> \${memoryUsage.modelSizeMB} MB<br>
                    <strong>Inference Activation Memory:</strong> \${memoryUsage.inferenceActivationSizeMB} MB<br>
                    <strong>Training Activation Memory:</strong> \${memoryUsage.trainingActivationSizeMB} MB
                </div>
                
                <h4>Model Details</h4>
                <ul>
                    <li><strong>Parameters:</strong> \${memoryUsage.details.parameters}</li>
                    <li><strong>Parameters (B):</strong> \${memoryUsage.details.parametersBillions}</li>
                    <li><strong>Hidden Size (D):</strong> \${memoryUsage.details.hiddenSize}</li>
                    <li><strong>Layers (L):</strong> \${memoryUsage.details.numLayers}</li>
                    <li><strong>Attention Heads (h):</strong> \${memoryUsage.details.numHeads}</li>
                    <li><strong>KV Heads (k):</strong> \${memoryUsage.details.numKVHeads}</li>
                    <li><strong>KV-to-Q Ratio (r):</strong> \${memoryUsage.details.kvRatio}</li>
                    <li><strong>FFN Dimension (F):</strong> \${memoryUsage.details.ffnSize}</li>
                    <li><strong>Vocabulary Size (V):</strong> \${memoryUsage.details.vocabSize}</li>
                    <li><strong>Quantization:</strong> \${memoryUsage.details.quantization} (\${memoryUsage.details.bytesPerParam} bytes/param)</li>
                    <li><strong>Batch Size (B):</strong> \${memoryUsage.details.batchSize}</li>
                    <li><strong>Sequence Length (T):</strong> \${memoryUsage.details.sequenceLength}</li>
                </ul>

                <h4>Formulas Used</h4>
                <div style="background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 5px; font-family: monospace;">
                    <p><strong>1. Total Parameters:</strong> P = V×D + L×[(2+r)D² + 3DF]</p>
                    <ul style="list-style-type: disc; padding-left: 20px;">
                        <li>The (2+r)D² term comes from the attention mechanism (Q, K, V and output for GQA)</li>
                        <li>The 3DF term is the feed-forward ("MLP") (first and second linear layers for GLU variants)</li>
                    </ul>
                    
                    <p><strong>2. Memory Footprint Estimates:</strong></p>
                    <p>- <strong>Weights memory:</strong> M<sub>weights</sub> = P × b (bytes)</p>
                    <p style="padding-left: 15px;">where b is bytes per parameter (\${memoryUsage.details.bytesPerParam} for \${memoryUsage.details.quantization})</p>
                    
                    <p>- <strong>Inference Activation memory:</strong> M<sub>act</sub> = B × T × D × b (bytes)</p>
                    <p style="padding-left: 15px;">where B = batch size, T = sequence length</p>
                    
                    <p>- <strong>Training Activation memory:</strong> M<sub>act</sub> = B × L × D × (T + 2D/h) × b (bytes)</p>
                    <p style="padding-left: 15px;">where B = batch size, L = number of layers, D = hidden size, h = number of heads</p>
                </div>
            \`;
            
            document.getElementById('memory-usage').innerHTML = memoryHtml;
            
            // Display model configuration
            const modelInfo = document.getElementById('model-info');
            modelInfo.innerHTML = '<h4>Full Model Configuration</h4><pre>' + 
                JSON.stringify(config, null, 2) + '</pre>';
        }
    </script>
</body>
</html>`;
    }
}

// This method is called when your extension is deactivated
export function deactivate() {}
