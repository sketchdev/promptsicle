import { EdgeHistoryItem, EdgeOptimizer, Item } from "@/types.ts";
import { traceLog } from "@/utils.ts";
import z from "zod";
import { structured } from "./llm.ts";

export const llmOptimizer =
  (goal: string, scoreTechnique: string, options: { instructions?: string; model?: string } = {}): EdgeOptimizer =>
  async (history: EdgeHistoryItem[], data: Item[]): Promise<string> => {
    traceLog("optimizer called");

    const bestScore = Math.max(...history.map((a) => a.score)); // best worst score
    const worstScore = Math.min(...history.map((a) => a.score)); // worst worst score

    traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: history.length });

    const defaultInstructions = `
your goal is to improve system instructions (llm prompt) for the given task based on past performance.
analyze the best attempts and determine what changes are needed to improve the prompt so that score is better than the best score so far.
use the provided examples of target outputs to understand how to change the prompt to achieve a higher score.
include real-world "few-shot" examples in the resulting prompt so the model can produce the desired output.
do not explain why the change was made.
respond with just the instruction text.`.trim();

    const previousAttempts = history.map((attempt) => {
      return `
<attempt>
  <instructions>${attempt.instructions}</instructions>
  <score>${attempt.score}</score>
  <prediction>${attempt.prediction}</prediction>
  <target>${attempt.target}</target>
</attempt>`;
    }).join("\n");

    const userPrompt = `
generate a new, improved system instruction that will achieve much a higher score than previous attempts. 

goal:
${goal}

scoring technique:
${scoreTechnique}

previous attempts:
${previousAttempts}

training data:
${JSON.stringify(data.slice(0, 3), null, 2)}`.trim();

    const response = await structured({
      model: options.model ?? "gpt-4o-mini",
      instructions: options.instructions ?? defaultInstructions,
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
