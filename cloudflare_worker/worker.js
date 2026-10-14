/**
 * Cloudflare Worker: Secure LLM Proxy for FreightTiger Assistant.
 * 
 * Holds the private LLM API key (e.g., GROQ_API_KEY) in encrypted Cloudflare Secrets.
 * Exposes a POST /generate endpoint that forwards prompts to Groq LLaMA-3.3-70B
 * and returns the generated text along with accurate token usage.
 */

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "healthy", model: "llama-3.3-70b-versatile" }), {
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
      const apiKey = env.GROQ_API_KEY;
      if (!apiKey) {
        return new Response(
          JSON.stringify({
            error: "GROQ_API_KEY secret is not configured on this Cloudflare Worker.",
          }),
          { status: 500, headers: { "Content-Type": "application/json" } }
        );
      }

      // Call Groq LLaMA-3.3-70B API
      const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: prompt },
          ],
          temperature: 0.0,
          max_tokens: 300,
        }),
      });

      if (!groqResponse.ok) {
        const errText = await groqResponse.text();
        return new Response(
          JSON.stringify({ error: "Groq API error", details: errText }),
          { status: groqResponse.status, headers: { "Content-Type": "application/json" } }
        );
      }

      const groqData = await groqResponse.json();
      const answer = groqData.choices?.[0]?.message?.content || "";
      const usage = groqData.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

      return new Response(
        JSON.stringify({
          output: answer.trim(),
          input_tokens: usage.prompt_tokens,
          output_tokens: usage.completion_tokens,
          model: "llama-3.3-70b-versatile",
          provider: "Groq via Cloudflare Worker",
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
