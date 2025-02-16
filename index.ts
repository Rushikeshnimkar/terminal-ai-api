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
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        const chatRequest: ChatRequest = {
            messages: [{
                role: 'user',
                content: prompt
            }],
            model: 'deepseek-ai/DeepSeek-V3',
            max_tokens: 512,
            temperature: 0.1,
            top_p: 0.9,
            stream: false
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
            console.error('API Error:', response.status, errorText);
            throw new Error(`API Error: ${response.status}`);
        }

        const result = await response.json();
        res.status(200).json(result);
    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
}