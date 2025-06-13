import { EdgePrompt, Item, Prompt, Proposer } from "@/types.ts";
import { traceLog } from "@/utils.ts";
import z from "zod";
import { structured } from "./llm.ts";

export const llmOptimizer =
  (goal: string, options: { model?: string } = {}) => async (prompts: EdgePrompt[], data: Item[]): Promise<string> => {
    traceLog("optimizer called");

    let systemPrompt = "";
    let userPrompt = "";

    const bestAttempts = (n: number) => {
      return prompts
        .sort((a, b) => b.score - a.score)
        .slice(0, n);
    };

    const bestScore = Math.max(...prompts.map((a) => a.score));
    const worstScore = Math.min(...prompts.map((a) => a.score));

    traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: prompts.length });

    systemPrompt = `
your goal is to improve system instructions (llm prompt) for the given task based on past performance.

analyze the best attempts and determine what changes are needed to improve the prompt so that score is better than the best score so far.
use the provided examples of target outputs to understand how to change the prompt to achieve a higher score.
include real-world "few-shot" examples in the resulting prompt so the model can produce the desired output.
do not explain why the change was made.
respond with just the instruction text.`.trim();

    userPrompt = `
generate a new, improved system instruction that will achieve much higher score than previous attempts. 

goal:
${goal}

best instructions so far:
${JSON.stringify(bestAttempts(3), null, 2)}

examples of target outputs:
${data.map((d) => d.target)}`.trim();

    const response = await structured({
      model: options.model ?? "gpt-4o-mini",
      instructions: systemPrompt,
      input: [{ role: "user" as const, content: userPrompt }],
      format: z.object({ new_instruction: z.string() }),
      formatName: "result",
    });

    if (!response || !response.new_instruction) {
      throw new Error("Failed to parse response from OpenAI");
    }

    const newInstruction = response.new_instruction;
    traceLog("new prompt generated", { instruction: newInstruction });

    return newInstruction;
  };

export function llmProposer<TStages extends string = string>(
  task: Record<TStages, string>,
  options: { model?: string } = {},
): Proposer<TStages> {
  return async ({ stageName, pastAttempts, dataSummary, initialPrompts }): Promise<Prompt> => {
    traceLog(`proposer called for stage: ${stageName}`, { pastAttemptsCount: pastAttempts.length });

    let systemPrompt = "";
    let userPrompt = "";

    if (pastAttempts.length === 0) {
      // first iteration - just return the base prompt
      return initialPrompts[stageName];
    }

    const bestAttempts = (n: number) => {
      return pastAttempts
        .sort((a, b) => b.score - a.score)
        .slice(0, n);
    };

    const bestScore = Math.max(...pastAttempts.map((a) => a.score));
    const worstScore = Math.min(...pastAttempts.map((a) => a.score));

    traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: pastAttempts.length });

    systemPrompt = `
your goal is to improve system instructions (prompts) for the given task (${task[stageName]}) based on past performance.

analyze the best and worst attempts and determine what changes are needed to improve the prompt so that score is better than the best score so far.
use the provided examples of target outputs to understand how to change the prompt to achieve a higher score.
include real-world "few-shot" examples in the resulting prompt so the model can produce the desired output.
do not explain why the change was made.
respond with just the instruction text.`.trim();

    userPrompt = `
generate a new, improved system instruction that will achieve much higher score than previous attempts. 

task: 
${task[stageName]}

best instructions so far:
${JSON.stringify(bestAttempts(3), null, 2)}

examples of target outputs:
${dataSummary}`.trim();

    const response = await structured({
      model: options.model ?? "gpt-4o-mini",
      instructions: systemPrompt,
      input: [{ role: "user" as const, content: userPrompt }],
      format: z.object({ new_instruction: z.string() }),
      formatName: "result",
      // temperature: Math.max(0.1, 1 - pastAttempts.length * 0.1), // anneal learning rate
    });

    if (!response || !response.new_instruction) {
      throw new Error("Failed to parse response from OpenAI");
    }

    const newInstruction = response.new_instruction;
    traceLog("new prompt generated", { stage: stageName, instruction: newInstruction });

    return {
      instruction: newInstruction,
      examples: [],
    };
  };
}
