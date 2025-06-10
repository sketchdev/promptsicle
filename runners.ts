import { create } from "@/llm.ts";
import { Item, PipelineOutput, Prompt } from "@/types.ts";
import { traceLog } from "@/utils.ts";

export function singleStageRunner(params: { model?: string; temperature?: number } = {}) {
  return async function (item: Item, prompts: Record<"generate", Prompt>): Promise<PipelineOutput> {
    traceLog(`pipeline running for item: "${item.text}"`);

    const { instruction, examples } = prompts.generate;

    const resp = await create({
      model: params.model ?? "gpt-4o-mini",
      instructions: instruction,
      input: [
        ...examples.flatMap((e) => [
          { role: "user" as const, content: e.input },
          { role: "assistant" as const, content: e.output },
        ]),
        { role: "user" as const, content: item.text },
      ],
      temperature: params.temperature ?? 0,
    });

    const predicted = resp.trim().toLowerCase();

    return { predicted, target: item.target };
  };
}
