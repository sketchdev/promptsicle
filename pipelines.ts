import { create } from "@/llm.ts";
import { Item, Pipeline, PipelineOutput, Prompt } from "@/types.ts";
import { promptBuilder, traceLog } from "@/utils.ts";

type SingleStage = "generate";

export function singleStagePromptBuilder(
  instruction: string,
  examples?: Record<string, unknown>[],
): Record<SingleStage, Prompt> {
  return {
    generate: promptBuilder(instruction, examples),
  };
}

export class SingleStagePipeline implements Pipeline<PipelineOutput, SingleStage> {
  private model: string;
  private temperature: number;

  constructor(params: { model?: string; temperature?: number } = {}) {
    this.model = params.model ?? "gpt-4o-mini";
    this.temperature = params.temperature ?? 0;
  }

  stageNames(): readonly SingleStage[] {
    return ["generate"] as const;
  }

  async run(item: Item, prompts: Record<SingleStage, Prompt>): Promise<PipelineOutput> {
    traceLog(`pipeline running for item: "${item.text}"`);

    const { instruction, examples } = prompts.generate;

    const resp = await create({
      model: this.model,
      instructions: instruction,
      input: [
        ...examples.flatMap((e) => [
          { role: "user" as const, content: e.input },
          { role: "assistant" as const, content: e.output },
        ]),
        { role: "user" as const, content: item.text },
      ],
      temperature: this.temperature,
    });

    const predicted = resp.trim().toLowerCase();

    return { predicted, target: item.target };
  }
}
