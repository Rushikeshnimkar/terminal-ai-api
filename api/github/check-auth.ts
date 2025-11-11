// api/github/check-auth.ts

import { type NextApiRequest, type NextApiResponse } from "next";

// Get these from your Vercel Environment Variables
const GITHUB_CLIENT_ID = process.env.GITHUB_CLIENT_ID;
const GITHUB_CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Allow CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { code: device_code } = req.query;

  if (!device_code) {
    return res.status(400).json({ error: "Missing 'code' parameter" });
  }

  if (!GITHUB_CLIENT_ID || !GITHUB_CLIENT_SECRET) {
    return res
      .status(500)
      .json({ error: "Server misconfigured: GitHub secrets not set." });
  }

  try {
    // 1. Poll GitHub to see if the user has authorized
    const githubResponse = await fetch(
      "https://github.com/login/oauth/access_token",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          client_id: GITHUB_CLIENT_ID,
          client_secret: GITHUB_CLIENT_SECRET,
          device_code: device_code as string,
          grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        }),
      }
    );

    if (!githubResponse.ok) {
      throw new Error(`GitHub token API error: ${await githubResponse.text()}`);
    }

    const data = await githubResponse.json();

    // 2. Check the response from GitHub
    if (data.error) {
      if (data.error === "authorization_pending") {
        // User hasn't finished yet. Tell the CLI to keep polling.
        return res.status(202).json({ status: "pending" });
      } else {
        // Any other error (e.g., 'expired_token')
        throw new Error(data.error_description);
      }
    }

    // 3. SUCCESS! We have the token.
    if (data.access_token) {
      // Send the token back to the CLI.
      return res.status(200).json({ token: data.access_token });
    }

    // Fallback for unexpected case
    return res.status(500).json({ error: "Unknown GitHub response" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return res.status(500).json({ error: message });
  }
}
