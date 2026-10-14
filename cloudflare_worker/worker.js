/**
 * Cloudflare Worker: Secure Gemini LLM Proxy for FreightTiger Assistant.
 * 
 * Holds the private GEMINI_API_KEY in encrypted Cloudflare Secrets.
 * Exposes a POST /generate endpoint that forwards prompts to Google Gemini (e.g. gemini-2.5-flash / gemini-1.5-flash)
 * and returns the generated text along with exact token usage from Gemini metadata.
 */

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "healthy", model: "gemini-flash-latest", provider: "Google Gemini via Cloudflare Worker" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (request.method !== "POST" || url.pathname !== "/generate") {
      return new Response(JSON.stringify({ error: "Endpoint not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    try {
      const body = await request.json();
      const prompt = body.prompt;
      const systemPrompt = body.system_prompt || "You are an expert shipping cost analyst for FreightTiger.";

      if (!prompt) {
        return new Response(JSON.stringify({ error: "Missing 'prompt' in request body" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }

      // Check if secret key is configured in Cloudflare environment
      const apiKey = env.GEMINI_API_KEY;
      if (!apiKey) {
        return new Response(
          JSON.stringify({
            error: "GEMINI_API_KEY secret is not configured on this Cloudflare Worker. Run 'npx wrangler secret put GEMINI_API_KEY'.",
          }),
          { status: 500, headers: { "Content-Type": "application/json" } }
        );
      }

      // Call Google Gemini REST API v1beta
      const modelName = env.GEMINI_MODEL || "gemini-3.5-flash-lite";
      const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey.trim()}`;

      const geminiPayload = {
        system_instruction: {
          parts: [{ text: systemPrompt }]
        },
        contents: [
          {
            role: "user",
            parts: [{ text: prompt }]
          }
        ],
        generationConfig: {
          temperature: 0.0,
          maxOutputTokens: 300,
        }
      };

      const geminiResponse = await fetch(geminiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(geminiPayload),
      });

      if (!geminiResponse.ok) {
        const errText = await geminiResponse.text();
        return new Response(
          JSON.stringify({ error: "Gemini API error", details: errText }),
          { status: geminiResponse.status, headers: { "Content-Type": "application/json" } }
        );
      }

      const geminiData = await geminiResponse.json();
      const answer = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || "";
      const usage = geminiData.usageMetadata || { promptTokenCount: 0, candidatesTokenCount: 0, totalTokenCount: 0 };

      return new Response(
        JSON.stringify({
          output: answer.trim(),
          input_tokens: usage.promptTokenCount,
          output_tokens: usage.candidatesTokenCount,
          model: modelName,
          provider: "Google Gemini via Cloudflare Worker",
        }),
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        }
      );
    } catch (err) {
      return new Response(
        JSON.stringify({ error: "Internal server error", message: err.message }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }
  },
};
