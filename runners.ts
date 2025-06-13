import { create } from "@/llm.ts";
import { EdgePrompt, Item, Prompt, Runner, RunnerOutput } from "@/types.ts";
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

export function singleStageRunner(
  params: { model?: string; temperature?: number } = {},
): Runner<RunnerOutput, "generate"> {
  return async function (item: Item, prompts: Record<"generate", Prompt>): Promise<RunnerOutput> {
    traceLog(`running for item: "${item.text.slice(0, 50)}..."`);

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

export function concatRunner(
  stages: string[],
  params: { model?: string; temperature?: number } = {},
): Runner<RunnerOutput, typeof stages[number]> {
  return async function (item: Item, prompts: Record<string, Prompt>): Promise<RunnerOutput> {
    traceLog(`running for item: "${item.text.slice(0, 50)}..." with stages: ${stages.join(", ")}`);

    const stagePromises = stages.map(async (stage, index) => {
      traceLog(`running stage: ${stage}`);
      const prompt = prompts[stage];
      const resp = await create({
        model: params.model ?? "gpt-4o-mini",
        instructions: prompt.instruction,
        input: [
          ...prompt.examples.flatMap((e) => [
            { role: "user" as const, content: e.input },
            { role: "assistant" as const, content: e.output },
          ]),
          { role: "user" as const, content: item.text },
        ],
        temperature: params.temperature ?? 0,
      });

      return { index, output: resp.trim() };
    });

    const results = await Promise.all(stagePromises);

    const output = results
      .sort((a, b) => a.index - b.index)
      .map((result) => result.output);

    return { predicted: output.join("\n"), target: item.target };
  };
}
