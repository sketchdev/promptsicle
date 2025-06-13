import { create } from "@/llm.ts";
import { EdgePrompt, Item } from "@/types.ts";
import { traceLog } from "@/utils.ts";

export const edgeRunner = (params: { model?: string; temperature?: number } = {}) =>
async (
  item: Item,
  prompt: EdgePrompt,
): Promise<string> => {
  traceLog(`running for item: "${item.text.slice(0, 50)}..."`);

  const resp = await create({
    model: params.model ?? "gpt-4o-mini",
    instructions: prompt.instructions,
    input: [
      { role: "user" as const, content: item.text },
    ],
    temperature: params.temperature ?? 0,
  });

  return resp.trim().toLowerCase();
};
