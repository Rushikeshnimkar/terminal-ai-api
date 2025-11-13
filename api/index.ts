// api/index.ts
import { type NextApiRequest, type NextApiResponse } from "next";
import { v4 as uuidv4 } from "uuid";
import { Resend } from "resend";

// Types
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

// Config
const VECTOR_DIMENSION = 1024;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || "";
const PINECONE_INDEX_NAME =
  process.env.PINECONE_INDEX_NAME || "terminal-ai-conversations";
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || "gcp-starter";

// --- Resend Configuration ---
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const NOTIFICATION_EMAIL_TO = process.env.NOTIFICATION_EMAIL_TO || "";
const NOTIFICATION_EMAIL_FROM =
  process.env.NOTIFICATION_EMAIL_FROM || "onboarding@resend.dev";

const resend = new Resend(RESEND_API_KEY);
// --- End of Resend Config ---

// --- NEW: Helper function with HEAVY LOGGING ---
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
// --- End of Email Function ---

// ... (All your Pinecone functions: pineconeListIndexes, pineconeCreateIndex, pineconeUpsert, pineconeQuery)
async function pineconeListIndexes(): Promise<string[]> {
  return [PINECONE_INDEX_NAME];
}
async function pineconeCreateIndex(indexName: string): Promise<boolean> {
  console.log(`Index ${indexName} is already available`);
  return true;
}
async function pineconeUpsert(
  indexName: string,
  vectors: PineconeVector[],
  namespace: string
): Promise<boolean> {
  try {
    const response = await fetch(
      "https://terminal-ai-conversations-yisuhd1.svc.aped-4627-b74a.pinecone.io/vectors/upsert",
      {
        method: "POST",
        headers: {
          "Api-Key": PINECONE_API_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ vectors, namespace }),
      }
    );
    if (!response.ok)
      throw new Error(`Pinecone upsert error: ${response.status}`);
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
    if (!response.ok)
      throw new Error(`Pinecone query error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    return { matches: [] };
  }
}

// ... (initPinecone, generateEmbedding, chunkText, saveConversation, getConversationHistory)
const initPinecone = async (): Promise<boolean> => {
  if (!PINECONE_API_KEY) {
    console.log("Pinecone API key not set, skipping initialization");
    return false;
  }
  console.log(`Using existing Pinecone index: ${PINECONE_INDEX_NAME}`);
  return true;
};
const generateEmbedding = (text: string): number[] => {
  try {
    const values = Array(VECTOR_DIMENSION).fill(0);
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const position = i % VECTOR_DIMENSION;
      values[position] += (charCode / 255) * Math.sin(i);
      const nextPos = (position + 1) % VECTOR_DIMENSION;
      const prevPos = (position - 1 + VECTOR_DIMENSION) % VECTOR_DIMENSION;
      values[nextPos] += (charCode / 510) * Math.cos(i);
      values[prevPos] += (charCode / 510) * Math.tan(i % 1.5);
    }
    const magnitude = Math.sqrt(
      values.reduce((sum, val) => sum + val * val, 0)
    );
    return values.map((val) => (magnitude > 0 ? val / magnitude : 0));
  } catch (error) {
    console.error("Error generating embedding:", error);
    return Array(VECTOR_DIMENSION)
      .fill(0)
      .map((_, i) => Math.sin(i));
  }
};
function chunkText(text: string, maxChunkSize: number = 512): string[] {
  const chunks: string[] = [];
  const sections = text.split(/\n\n+/);
  let currentChunk = "";
  for (const section of sections) {
    if ((currentChunk + section).length <= maxChunkSize) {
      currentChunk += (currentChunk ? "\n\n" : "") + section;
    } else {
      if (currentChunk) chunks.push(currentChunk);
      if (section.length > maxChunkSize) {
        const sentences = section.split(/(?<=[.!?])\s+/);
        let sectionChunk = "";
        for (const sentence of sentences) {
          if ((sectionChunk + sentence).length <= maxChunkSize) {
            sectionChunk += (sectionChunk ? " " : "") + sentence;
          } else {
            if (sectionChunk) chunks.push(sectionChunk);
            sectionChunk = sentence;
          }
        }
        if (sectionChunk) chunks.push(sectionChunk);
      } else {
        currentChunk = section;
      }
    }
  }
  if (currentChunk) chunks.push(currentChunk);
  return chunks;
}
const saveConversation = async (
  conversationId: string,
  userInput: string,
  aiResponse: string
): Promise<boolean> => {
  try {
    const currentTime = Date.now();
    const vectors: PineconeVector[] = [];
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
const getConversationHistory = async (
  conversationId: string
): Promise<ChatMessage[]> => {
  try {
    const queryResponse = await pineconeQuery(
      PINECONE_INDEX_NAME,
      { conversationId },
      100,
      "conversations"
    );
    if (!queryResponse.matches || queryResponse.matches.length === 0) return [];
    const messageMap = new Map<string, any[]>();
    for (const match of queryResponse.matches) {
      if (!match.metadata || !match.metadata.role || !match.metadata.content)
        continue;
      const key = `${match.metadata.role}-${match.metadata.timestamp}`;
      if (!messageMap.has(key)) messageMap.set(key, []);
      messageMap.get(key)!.push(match.metadata);
    }
    const messages: ChatMessage[] = [];
    for (const [key, chunks] of messageMap.entries()) {
      chunks.sort((a, b) => (a.chunkIndex || 0) - (b.chunkIndex || 0));
      const content = chunks.map((chunk) => chunk.content).join(" ");
      const [role, timestamp] = key.split("-");
      messages.push({
        role: role as "user" | "assistant",
        content,
        timestamp: parseInt(timestamp),
      });
    }
    return messages.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  } catch (error) {
    console.error("Error retrieving from Pinecone:", error);
    return [];
  }
};

// ... (createCommandSystemPrompt, createChatSystemPrompt)
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
  return `Task: Analyze the user's request, formulate a step-by-step reasoning plan...
...
User request: ${userPrompt}
...
Your JSON response:`; // (Keeping this short, your prompt is fine)
};
const createChatSystemPrompt = (
  userPrompt: string,
  history: ChatMessage[] = []
): { role: "user" | "assistant"; content: string }[] => {
  const systemMessage = {
    role: "assistant" as const,
    content: `You are T-AI, a helpful AI assistant...`, // (Keeping this short)
  };
  const historyMessages = history.slice(-10).map((msg) => ({
    role: msg.role,
    content: msg.content,
  }));
  const messages = [
    systemMessage,
    ...historyMessages,
    { role: "user" as const, content: userPrompt },
  ];
  return messages;
};

// --- Main Handler ---
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method === "OPTIONS") {
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
    return res.status(200).end();
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const pineconeInitialized = await initPinecone();
    console.log("Pinecone initialized:", pineconeInitialized);
    const body = req.body;
    const userPrompt = body.prompt;
    const mode = body.mode || "command";
    const conversationId = body.conversationId || uuidv4();
    console.log(`Received request for mode: ${mode}`);
    console.log("Received user prompt:", userPrompt);
    if (!userPrompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }
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

    let model: string;
    let messages: { role: string; content: string }[];
    let temperature: number;

    // --- USING MINIMAX FOR TESTING AS REQUESTED ---
    if (mode === "chat") {
      console.log("Using CHAT mode");
      model = "openrouter/polaris-alpha";
      messages = createChatSystemPrompt(userPrompt, history);
      temperature = 0.7;
    } else {
      console.log("Using COMMAND mode");
      model = "openrouter/polaris-alpha";
      const systemPrompt = createCommandSystemPrompt(userPrompt, history);
      messages = [{ role: "user", content: systemPrompt }];
      temperature = 0.3;
    }
    // --- END OF TEST MODIFICATION ---

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 55000);

    try {
      console.log(`Sending request to AI model: ${model}`);
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

      // ... (rest of the try block)
      const data = await response.json();
      console.log("Received response from AI model");
      if (pineconeInitialized && data?.choices?.[0]?.message?.content) {
        const aiResponse = data.choices[0].message.content;
        saveConversation(conversationId, userPrompt, aiResponse).catch((err) =>
          console.error("Non-blocking save failed:", err)
        );
      }
      const enhancedResponse = { ...data, conversationId };
      res.setHeader("Access-Control-Allow-Origin", "*");
      return res.status(200).json(enhancedResponse);
    } catch (error) {
      clearTimeout(timeout);
      throw error;
    }
  } catch (error) {
    console.error("API Error:", error);

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";

    if (error instanceof Error && error.name === "AbortError") {
      return res.status(504).json({ error: "Request timeout" });
    }

    // --- NEW: Email Logic with await ---
    const errorString = errorMessage.toLowerCase();
    if (
      errorString.includes("api error") ||
      errorString.includes("model not found") ||
      errorString.includes("404") || // This will match your 404
      errorString.includes("500") ||
      errorString.includes("503") ||
      errorString.includes("401")
    ) {
      console.log(
        "--- [DEBUG] Error matched. AWAITING sendFailureEmail()... ---"
      );

      // THIS IS THE FIX: We added "await" here.
      try {
        await sendFailureEmail(errorMessage, JSON.stringify(error, null, 2));
      } catch (emailErr) {
        console.error(
          "--- [DEBUG] sendFailureEmail function itself failed:",
          emailErr
        );
      }
    }
    // --- End of Email Logic ---

    // This line will now only run AFTER the email function has finished.
    return res.status(500).json({
      error: "Internal server error",
      details: errorMessage,
    });
  }
}
