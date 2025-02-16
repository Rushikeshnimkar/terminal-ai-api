import type { VercelRequest, VercelResponse } from '@vercel/node';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

interface ChatRequest {
    messages: ChatMessage[];
    model: string;
    max_tokens: number;
    temperature: number;
    top_p: number;
    stream: boolean;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    // Handle OPTIONS request
    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { messages, model, max_tokens, temperature, top_p, stream } = req.body;
        
        if (!messages) {
            return res.status(400).json({ error: 'Messages are required' });
        }

        const chatRequest: ChatRequest = {
            messages,
            model: model || 'deepseek-ai/DeepSeek-V3',
            max_tokens: max_tokens || 512,
            temperature: temperature || 0.1,
            top_p: top_p || 0.9,
            stream: stream || false
        };

        const response = await fetch('https://api.hyperbolic.xyz/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.HYPERBOLIC_API_KEY}`,
            },
            body: JSON.stringify(chatRequest)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Hyperbolic API Error:', response.status, errorText);
            throw new Error(`Hyperbolic API Error: ${response.status}`);
        }

        const result = await response.json();
        res.status(200).json(result);
    } catch (error) {
        console.error('Server Error:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            details: error instanceof Error ? error.message : 'Unknown error'
        });
    }
}