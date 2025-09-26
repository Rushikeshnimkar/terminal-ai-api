import { type NextRequest } from "next/server";
import { v4 as uuidv4 } from "uuid";

// Types for better type safety
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: number;
}

interface PineconeMetadata {
  conversationId: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  chunkIndex?: number;
  totalChunks?: number;
  source?: string;
}

interface PineconeVector {
  id: string;
  values: number[];
  metadata: PineconeMetadata;
}

// Configuration
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY = 1000;
const VECTOR_DIMENSION = 1024;

// Use environment variables to configure Pinecone
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || "";
const PINECONE_INDEX_NAME =
  process.env.PINECONE_INDEX_NAME || "terminal-ai-conversations";
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || "gcp-starter";

// Direct HTTP calls to Pinecone instead of using the SDK
async function pineconeListIndexes(): Promise<string[]> {
  // Since we already know your index exists, we can skip listing
  // and just return a hardcoded array with your index name
  return [PINECONE_INDEX_NAME];
}

async function pineconeCreateIndex(indexName: string): Promise<boolean> {
  // Since we know the index already exists, just return true
  console.log(`Index ${indexName} is already available`);
  return true;
}

async function pineconeUpsert(
  indexName: string,
  vectors: PineconeVector[],
  namespace: string
): Promise<boolean> {
  try {
    // Use the exact hostname provided in your Pinecone details
    const response = await fetch(
      "https://terminal-ai-conversations-yisuhd1.svc.aped-4627-b74a.pinecone.io/vectors/upsert",
      {
        method: "POST",
        headers: {
          "Api-Key": PINECONE_API_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          vectors,
          namespace,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Pinecone upsert error: ${response.status}`);
    }

    return true;
  } catch (error) {
    console.error("Error upserting vectors to Pinecone:", error);
    return false;
  }
}

async function pineconeQuery(
  indexName: string,
  filter: any,
  topK: number,
  namespace: string
): Promise<any> {
  try {
    // Use the exact hostname provided in your Pinecone details
    const response = await fetch(
      "https://terminal-ai-conversations-yisuhd1.svc.aped-4627-b74a.pinecone.io/query",
      {
        method: "POST",
        headers: {
          "Api-Key": PINECONE_API_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          vector: Array(VECTOR_DIMENSION).fill(0),
          filter,
          topK,
          includeMetadata: true,
          namespace,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Pinecone query error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    return { matches: [] };
  }
}

// Initialize Pinecone
const initPinecone = async (): Promise<boolean> => {
  // Skip Pinecone if API key is not set
  if (!PINECONE_API_KEY) {
    console.log("Pinecone API key not set, skipping initialization");
    return false;
  }

  // We already know the index exists, so just return true
  console.log(`Using existing Pinecone index: ${PINECONE_INDEX_NAME}`);
  return true;
};

// Generate embeddings - Edge-compatible version using simple math
const generateEmbedding = (text: string): number[] => {
  try {
    const values = Array(VECTOR_DIMENSION).fill(0);

    // Simple character-based hashing to generate values
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const position = i % VECTOR_DIMENSION;

      // Use character codes to create pseudo-random values
      values[position] += (charCode / 255) * Math.sin(i);

      // Add some values based on neighboring positions to increase complexity
      const nextPos = (position + 1) % VECTOR_DIMENSION;
      const prevPos = (position - 1 + VECTOR_DIMENSION) % VECTOR_DIMENSION;
      values[nextPos] += (charCode / 510) * Math.cos(i);
      values[prevPos] += (charCode / 510) * Math.tan(i % 1.5);
    }

    // Normalize the vector
    const magnitude = Math.sqrt(
      values.reduce((sum, val) => sum + val * val, 0)
    );
    return values.map((val) => (magnitude > 0 ? val / magnitude : 0));
  } catch (error) {
    console.error("Error generating embedding:", error);
    // Return a deterministic but still varying embedding as fallback
    return Array(VECTOR_DIMENSION)
      .fill(0)
      .map((_, i) => Math.sin(i));
  }
};

// Process and chunk text for better semantic representation
function chunkText(text: string, maxChunkSize: number = 512): string[] {
  // Simple chunking implementation
  const chunks: string[] = [];

  // Split by sections first (double newlines)
  const sections = text.split(/\n\n+/);

  let currentChunk = "";

  for (const section of sections) {
    // If adding this section doesn't exceed chunk size
    if ((currentChunk + section).length <= maxChunkSize) {
      currentChunk += (currentChunk ? "\n\n" : "") + section;
    } else {
      // Add current chunk if not empty
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      // Handle large sections
      if (section.length > maxChunkSize) {
        // Split by sentences
        const sentences = section.split(/(?<=[.!?])\s+/);
        let sectionChunk = "";

        for (const sentence of sentences) {
          if ((sectionChunk + sentence).length <= maxChunkSize) {
            sectionChunk += (sectionChunk ? " " : "") + sentence;
          } else {
            if (sectionChunk) {
              chunks.push(sectionChunk);
            }
            sectionChunk = sentence;
          }
        }

        if (sectionChunk) {
          chunks.push(sectionChunk);
        }
      } else {
        // Start a new chunk with current section
        currentChunk = section;
      }
    }
  }

  // Add the last chunk if not empty
  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

// Save conversation to Pinecone with optimized vector storage
const saveConversation = async (
  conversationId: string,
  userInput: string,
  aiResponse: string
): Promise<boolean> => {
  try {
    const currentTime = Date.now();
    const vectors: PineconeVector[] = [];

    // Process user input
    const userChunks = chunkText(userInput);
    for (let i = 0; i < userChunks.length; i++) {
      const chunk = userChunks[i];
      const embedding = generateEmbedding(chunk);
      vectors.push({
        id: `${conversationId}-user-${currentTime}-${i}`,
        values: embedding,
        metadata: {
          conversationId,
          role: "user",
          content: chunk,
          timestamp: currentTime,
          chunkIndex: i,
          totalChunks: userChunks.length,
          source: "user-message",
        },
      });
    }

    // Process AI response
    const aiChunks = chunkText(aiResponse);
    for (let i = 0; i < aiChunks.length; i++) {
      const chunk = aiChunks[i];
      const embedding = generateEmbedding(chunk);
      vectors.push({
        id: `${conversationId}-assistant-${currentTime + 1}-${i}`,
        values: embedding,
        metadata: {
          conversationId,
          role: "assistant",
          content: chunk,
          timestamp: currentTime + 1,
          chunkIndex: i,
          totalChunks: aiChunks.length,
          source: "assistant-message",
        },
      });
    }

    // Upload in batches
    if (vectors.length > 0) {
      const BATCH_SIZE = 100;
      for (let i = 0; i < vectors.length; i += BATCH_SIZE) {
        const batch = vectors.slice(i, i + BATCH_SIZE);
        await pineconeUpsert(PINECONE_INDEX_NAME, batch, "conversations");
      }
    }

    console.log(`Saved ${vectors.length} conversation vectors to Pinecone`);
    return true;
  } catch (error) {
    console.error("Error saving to Pinecone:", error);
    return false;
  }
};

// Retrieve conversation history with improved reconstruction
const getConversationHistory = async (
  conversationId: string
): Promise<ChatMessage[]> => {
  try {
    // Query for conversation messages
    const queryResponse = await pineconeQuery(
      PINECONE_INDEX_NAME,
      { conversationId },
      100,
      "conversations"
    );

    if (!queryResponse.matches || queryResponse.matches.length === 0) {
      return [];
    }

    // Group by role and timestamp to reconstruct messages
    const messageMap = new Map<string, any[]>();

    for (const match of queryResponse.matches) {
      if (!match.metadata || !match.metadata.role || !match.metadata.content)
        continue;

      const key = `${match.metadata.role}-${match.metadata.timestamp}`;
      if (!messageMap.has(key)) {
        messageMap.set(key, []);
      }
      messageMap.get(key)!.push(match.metadata);
    }

    // Reconstruct messages
    const messages: ChatMessage[] = [];

    for (const [key, chunks] of messageMap.entries()) {
      // Sort chunks by index
      chunks.sort((a, b) => (a.chunkIndex || 0) - (b.chunkIndex || 0));

      // Combine content
      const content = chunks.map((chunk) => chunk.content).join(" ");
      const [role, timestamp] = key.split("-");

      messages.push({
        role: role as "user" | "assistant",
        content,
        timestamp: parseInt(timestamp), // Store timestamp in the message
      });
    }

    // Sort by timestamp
    return messages.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  } catch (error) {
    console.error("Error retrieving from Pinecone:", error);
    return [];
  }
};

// Enhanced system prompt with more context awareness
const createSystemPrompt = (
  userPrompt: string,
  history: ChatMessage[] = []
): string => {
  const historyText =
    history.length > 0
      ? `Previous conversation:\n${history
          .map((msg) => `${msg.role}: ${msg.content}`)
          .join("\n")}`
      : "No previous conversation";

  return `Task: Generate a valid Windows Command
User request: ${userPrompt}

${historyText}

Requirements:

1. provide ONLY ONE single command without explanation.
2. Use relative paths where applicable.
3. When showing file content, check file existence first
4. When using environment variables, verify they exist

Your response:`;
};

export const config = {
  runtime: "edge",
};

export default async function handler(req: NextRequest) {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: {
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS,PATCH,DELETE,POST,PUT",
        "Access-Control-Allow-Headers":
          "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version",
      },
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    // Initialize Pinecone - but don't fail if it doesn't work
    const pineconeInitialized = await initPinecone();
    console.log("Pinecone initialized:", pineconeInitialized);

    const body = await req.json();
    const userPrompt = body.prompt || body.messages?.[0]?.content;
    const conversationId = body.conversationId || uuidv4();

    if (!userPrompt) {
      return new Response(JSON.stringify({ error: "Prompt is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Get history only if Pinecone is working
    let history: ChatMessage[] = [];
    if (pineconeInitialized) {
      try {
        history = await getConversationHistory(conversationId);
        console.log(
          `Retrieved ${history.length} messages from conversation history`
        );
      } catch (historyError) {
        console.error(
          "Error retrieving history, continuing without it:",
          historyError
        );
      }
    }

    // Generate system prompt with conversation history
    const systemPrompt = createSystemPrompt(userPrompt, history);

    const messages = [
      {
        role: "user",
        content: systemPrompt,
      },
    ];

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);

    try {
      console.log("Sending request to AI model");
      const response = await fetch(
        "https://openrouter.ai/api/v1/chat/completions",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
            "HTTP-Referer": "https://terminal-ai-api.vercel.app",
            "X-Title": "Terminal AI Assistant",
          },
          body: JSON.stringify({
            model: "x-ai/grok-4-fast:free",
            messages,
            temperature: 0.3,
            top_p: 0.9,
          }),
          signal: controller.signal,
        }
      );

      clearTimeout(timeout);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error (${response.status}): ${errorText}`);
      }

      const data = await response.json();
      console.log("Received response from AI model");

      // Save conversation to Pinecone
      if (pineconeInitialized && data?.choices?.[0]?.message?.content) {
        const aiResponse = data.choices[0].message.content;
        console.log("Saving conversation to Pinecone");
        await saveConversation(conversationId, userPrompt, aiResponse);
      }

      // Add conversationId to the response
      const enhancedResponse = {
        ...data,
        conversationId,
      };

      return new Response(JSON.stringify(enhancedResponse), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error) {
      clearTimeout(timeout);
      throw error;
    }
  } catch (error) {
    console.error("API Error:", error);

    if (error instanceof Error && error.name === "AbortError") {
      return new Response(JSON.stringify({ error: "Request timeout" }), {
        status: 504,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        error: "Internal server error",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
