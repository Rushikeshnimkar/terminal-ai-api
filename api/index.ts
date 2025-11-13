import { type NextApiRequest, type NextApiResponse } from "next";
import { v4 as uuidv4 } from "uuid";
import { Resend } from "resend"; // --- ADDED ---

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

// --- [ADDED] Resend Configuration ---
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const NOTIFICATION_EMAIL_TO = process.env.NOTIFICATION_EMAIL_TO || "";
const NOTIFICATION_EMAIL_FROM =
  process.env.NOTIFICATION_EMAIL_FROM || "onboarding@resend.dev";

const resend = new Resend(RESEND_API_KEY);
// --- [ADDED] End of Resend Config ---

// --- [ADDED] Helper function with HEAVY LOGGING ---
async function sendFailureEmail(errorMessage: string, errorDetails: string) {
  console.log("--- [DEBUG] INSIDE sendFailureEmail function. ---");

  // This is the most likely point of failure
  if (!RESEND_API_KEY || !NOTIFICATION_EMAIL_TO) {
    console.error(
      "--- [DEBUG] FAILED: Resend variables not set. Cannot send email. ---"
    );
    console.log(`--- [DEBUG] RESEND_API_KEY is set: ${!!RESEND_API_KEY}`);
    console.log(
      `--- [DEBUG] NOTIFICATION_EMAIL_TO is set: ${!!NOTIFICATION_EMAIL_TO}`
    );
    return;
  }

  try {
    console.log(
      `--- [DEBUG] Variables are set. Attempting to send email... ---`
    );
    console.log(`--- [DEBUG] From: ${NOTIFICATION_EMAIL_FROM}`);
    console.log(`--- [DEBUG] To: ${NOTIFICATION_EMAIL_TO}`);

    await resend.emails.send({
      from: NOTIFICATION_EMAIL_FROM,
      to: NOTIFICATION_EMAIL_TO,
      subject: "URGENT: Terminal AI Model Failure",
      html: `
        <h1>T-AI Model Alert</h1>
        <p>The API encountered a critical error when trying to reach the OpenRouter model.</p>
        <hr>
        <p><strong>Error Message:</strong></p>
        <pre>${errorMessage}</pre>
        <br>
        <p><strong>Full Error Details:</strong></p>
        <pre>${errorDetails}</pre>
      `,
    });

    console.log("--- [DEBUG] SUCCESS: Email sent via Resend. ---");
  } catch (emailError) {
    console.error(
      "--- [DEBUG] FAILED: Resend.emails.send() threw an error. ---"
    );
    console.error(emailError);
  }
}
// --- [ADDED] End of Email Function ---

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

// ... (initPinecone, generateEmbedding, chunkText, saveConversation, getConversationHistory)
// ...
const initPinecone = async (): Promise<boolean> => {
  // Skip Pinecone if API key is not set
  if (!PINECONE_API_KEY) {
    console.log("Pinecone API key not set, skipping initialization");
    return false;
  } // We already know the index exists, so just return true

  console.log(`Using existing Pinecone index: ${PINECONE_INDEX_NAME}`);
  return true;
};

// Generate embeddings - Edge-compatible version using simple math
const generateEmbedding = (text: string): number[] => {
  try {
    const values = Array(VECTOR_DIMENSION).fill(0); // Simple character-based hashing to generate values

    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const position = i % VECTOR_DIMENSION; // Use character codes to create pseudo-random values

      values[position] += (charCode / 255) * Math.sin(i); // Add some values based on neighboring positions to increase complexity

      const nextPos = (position + 1) % VECTOR_DIMENSION;
      const prevPos = (position - 1 + VECTOR_DIMENSION) % VECTOR_DIMENSION;
      values[nextPos] += (charCode / 510) * Math.cos(i);
      values[prevPos] += (charCode / 510) * Math.tan(i % 1.5);
    } // Normalize the vector

    const magnitude = Math.sqrt(
      values.reduce((sum, val) => sum + val * val, 0)
    );
    return values.map((val) => (magnitude > 0 ? val / magnitude : 0));
  } catch (error) {
    console.error("Error generating embedding:", error); // Return a deterministic but still varying embedding as fallback
    return Array(VECTOR_DIMENSION)
      .fill(0)
      .map((_, i) => Math.sin(i));
  }
};

// Process and chunk text for better semantic representation
function chunkText(text: string, maxChunkSize: number = 512): string[] {
  // Simple chunking implementation
  const chunks: string[] = []; // Split by sections first (double newlines)

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
      } // Handle large sections

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
  } // Add the last chunk if not empty

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
    const vectors: PineconeVector[] = []; // Process user input

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
    } // Process AI response

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
    } // Upload in batches

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
      100, // Get up to 100 message chunks
      "conversations"
    );

    if (!queryResponse.matches || queryResponse.matches.length === 0) {
      return [];
    } // Group by role and timestamp to reconstruct messages

    const messageMap = new Map<string, any[]>();

    for (const match of queryResponse.matches) {
      if (!match.metadata || !match.metadata.role || !match.metadata.content)
        continue;

      const key = `${match.metadata.role}-${match.metadata.timestamp}`;
      if (!messageMap.has(key)) {
        messageMap.set(key, []);
      }
      messageMap.get(key)!.push(match.metadata);
    } // Reconstruct messages

    const messages: ChatMessage[] = [];

    for (const [key, chunks] of messageMap.entries()) {
      // Sort chunks by index
      chunks.sort((a, b) => (a.chunkIndex || 0) - (b.chunkIndex || 0)); // Combine content

      const content = chunks.map((chunk) => chunk.content).join(" ");
      const [role, timestamp] = key.split("-");

      messages.push({
        role: role as "user" | "assistant",
        content,
        timestamp: parseInt(timestamp), // Store timestamp in the message
      });
    } // Sort by timestamp

    return messages.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  } catch (error) {
    console.error("Error retrieving from Pinecone:", error);
    return [];
  }
};

// ***********************************************
// *** 1. KEY CHANGE: RENAMED & UPDATED COMMAND PROMPT ***
// This is your *original* prompt for generating commands,
// but using the better JSON-requesting version from aiService.ts
// ***********************************************
const createCommandSystemPrompt = (
  userPrompt: string,
  history: ChatMessage[] = []
): string => {
  const historyText =
    history.length > 0
      ? `Previous conversation:\n${history
          .map((msg) => `${msg.role}: ${msg.content}`)
          .join("\n")}`
      : "No previous conversation";

  // This is the prompt from your aiService.ts, which is much better
  // and asks for the JSON format.
  return `Task: Analyze the user's request, formulate a step-by-step reasoning plan, and then generate a single, valid Windows Command Prompt (CMD) command to accomplish it.

System Information:
Current directory: (User's CWD)
OS: Windows

${historyText ? `Recent conversation:\n${historyText}\n\n` : ""}

User request: ${userPrompt}

Requirements:
1.  **Reasoning:** First, provide a brief, step-by-step plan (as a string) explaining how you'll achieve the user's request.
2.  **Command:** Second, provide ONLY ONE single-line, executable CMD command. No PowerShell.
3.  **Safety:** Avoid destructive commands unless explicitly asked. Use relative paths.
4.  **Format:** Your response MUST be in this exact JSON format:
    {
      "reasoning": "Your step-by-step plan here.",
      "command": "Your single-line command here."
    }

Your JSON response:`;
};

// ***********************************************
// *** 2. KEY CHANGE: ADD NEW CHAT PROMPT FUNCTION ***
// This is the *new* prompt for conversational chat.
// ***********************************************
const createChatSystemPrompt = (
  userPrompt: string,
  history: ChatMessage[] = []
): { role: "user" | "assistant"; content: string }[] => {
  // Create the system message
  const systemMessage = {
    role: "assistant" as const,
    content: `You are T-AI, a helpful AI assistant operating in a terminal.
Provide clear, concise, and well-formatted answers.
Use markdown for formatting, especially for code blocks.
You are not a command generator; you are a conversational assistant.`,
  };

  // Format history for the chat model (get last 10 messages)
  const historyMessages = history.slice(-10).map((msg) => ({
    role: msg.role,
    content: msg.content,
  }));

  // Construct the final message array
  const messages = [
    systemMessage,
    ...historyMessages,
    { role: "user" as const, content: userPrompt },
  ];

  return messages;
};

// export const config = {
//   runtime: "edge",
// };

// ✅ CHANGED: Updated signature to use NextApiRequest and NextApiResponse
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method === "OPTIONS") {
    // ✅ CHANGED: Set headers on the response object
    res.setHeader("Access-Control-Allow-Credentials", "true");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader(
      "Access-Control-Allow-Methods",
      "GET,OPTIONS,PATCH,DELETE,POST,PUT"
    );
    res.setHeader(
      "Access-Control-Allow-Headers",
      "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version"
    );
    return res.status(200).end(); // Send empty response
  }

  if (req.method !== "POST") {
    // ✅ CHANGED: Use res object for response
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    // Initialize Pinecone - but don't fail if it doesn't work
    const pineconeInitialized = await initPinecone();
    console.log("Pinecone initialized:", pineconeInitialized);

    // ✅ CHANGED: Access pre-parsed body
    const body = req.body;

    // *****************************************************************
    // *** 3. KEY CHANGE: GET PROMPT AND MODE FROM REQUEST BODY ***
    // *****************************************************************
    const userPrompt = body.prompt; // We will now get the raw user input
    const mode = body.mode || "command"; // Default to "command" for backward compatibility
    const conversationId = body.conversationId || uuidv4();

    console.log(`Received request for mode: ${mode}`);

    // ✅ Your log will now work!
    console.log("Received user prompt:", userPrompt);

    if (!userPrompt) {
      // ✅ CHANGED: Use res object for response
      return res.status(400).json({ error: "Prompt is required" });
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

    // *****************************************************************
    // *** 4. KEY CHANGE: ROUTE BASED ON MODE ***
    // *****************************************************************

    let model: string;
    let messages: { role: string; content: string }[];
    let temperature: number;

    if (mode === "chat") {
      // --- CHAT MODE ---
      console.log("Using CHAT mode");
      model = "minimax/minimax-m2:free"; // Your chat model
      messages = createChatSystemPrompt(userPrompt, history);
      temperature = 0.7; // More creative for chat
    } else {
      // --- COMMAND MODE (default) ---
      console.log("Using COMMAND mode");
      model = "minimax/minimax-m2:free"; // Or your original command model
      const systemPrompt = createCommandSystemPrompt(userPrompt, history);
      messages = [
        {
          role: "user",
          content: systemPrompt,
        },
      ];
      temperature = 0.3; // Stricter for command generation
    }

    const controller = new AbortController();
    // ✅ UPDATED: Set timeout to 55 seconds (less than your 60s vercel.json)
    const timeout = setTimeout(() => controller.abort(), 55000);

    try {
      console.log(`Sending request to AI model: ${model}`);
      // --- [FIXED] ---
      // Removed the apostrophe from "https'://"
      const response = await fetch(
        "https://openrouter.ai/api/v1/chat/completions",
        // --- [END FIX] ---
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            // THE API KEY IS SECURELY STORED ON VERCEL
            Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
            "HTTP-Referer": "https://terminal-ai-api.vercel.app",
            "X-Title": "Terminal AI Assistant",
          },
          body: JSON.stringify({
            model: model,
            messages: messages,
            temperature: temperature,
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

        // Run this in the background (non-blocking)
        // We save the *raw* user prompt, not the system prompt
        saveConversation(conversationId, userPrompt, aiResponse).catch((err) =>
          console.error("Non-blocking save failed:", err)
        );
      }

      // Add conversationId to the response
      const enhancedResponse = {
        ...data,
        conversationId,
      };

      // ✅ CHANGED: Use res object for response
      res.setHeader("Access-Control-Allow-Origin", "*");
      return res.status(200).json(enhancedResponse);
    } catch (error) {
      clearTimeout(timeout);
      throw error;
    }
  } catch (error) {
    console.error("API Error:", error);

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error"; // --- [MODIFIED] ---

    if (error instanceof Error && error.name === "AbortError") {
      // ✅ CHANGED: Use res object for response
      return res.status(504).json({ error: "Request timeout" });
    }

    // --- [MODIFIED] Email Logic with await and full debug logging ---
    const errorString = errorMessage.toLowerCase();
    if (
      errorString.includes("api error") ||
      errorString.includes("model not found") ||
      errorString.includes("404") ||
      errorString.includes("500") ||
      errorString.includes("503") ||
      errorString.includes("401")
    ) {
      console.log(
        "--- [DEBUG] Error matched. AWAITING sendFailureEmail()... ---"
      );

      try {
        // THIS IS THE FIX: We added "await" here.
        await sendFailureEmail(errorMessage, JSON.stringify(error, null, 2));
      } catch (emailErr) {
        console.error(
          "--- [DEBUG] sendFailureEmail function itself failed:",
          emailErr
        );
      }
    }
    // --- [MODIFIED] End of Email Logic ---

    // ✅ CHANGED: Use res object for response
    return res.status(500).json({
      error: "Internal server error",
      details: errorMessage, // --- [MODIFIED] ---
    });
  }
}
