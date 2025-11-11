// api/github/start-auth.ts
import { type NextApiRequest, type NextApiResponse } from "next";

const GITHUB_CLIENT_ID = process.env.GITHUB_CLIENT_ID;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // ... (Your CORS and method checks are fine)
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!GITHUB_CLIENT_ID) {
    return res
      .status(500)
      .json({ error: "Server misconfigured: GITHUB_CLIENT_ID is not set." });
  }

  try {
    const githubResponse = await fetch("https://github.com/login/device/code", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        client_id: GITHUB_CLIENT_ID,
        scope: "repo read:user",
      }),
    });

    if (!githubResponse.ok) {
      throw new Error(`GitHub API error: ${await githubResponse.text()}`);
    }

    const data = await githubResponse.json();

    // ✅ **THE CHANGE IS HERE:**
    // We must send all these values back to the CLI
    return res.status(200).json({
      user_code: data.user_code,
      verification_uri: data.verification_uri,
      interval: data.interval,
      device_code: data.device_code, // <-- This is the secret
      expires_in: data.expires_in, // <-- This is the timeout
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return res.status(500).json({ error: message });
  }
}
