import { Prompt, Proposer } from "@/types.ts";
import { traceLog } from "@/utils.ts";
import z from "zod";
import { structured } from "./llm.ts";

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

    // analyze past attempts and generate improvements
    const performanceAnalysis = pastAttempts
      .map((attempt, i) => `attempt ${i + 1}: "${attempt.prompt.instruction}" (score: ${attempt.score.toFixed(3)})`)
      .join("\n");

    const bestScore = Math.max(...pastAttempts.map((a) => a.score));
    const worstScore = Math.min(...pastAttempts.map((a) => a.score));

    traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: pastAttempts.length });

    systemPrompt = `
your goal is to improve system instructions (prompts) for the given task (${task[stageName]}) based on past performance.
    
analyze past prompt performance and the provided output examples to try and determine what makes a good prompt for this task.
think about why previous prompts succeeded or failed, and how you can improve clarity, specificity, and handling of edge cases.
the goal is to achieve higher accuracy than previous attempts by generating a new prompt that produces content that is closer to matching a cosine similarity of 1.0 with the target content.
include few-shot examples so the model can produce the desired output.
focus on clarity, specificity, and handling edge cases. 
respond with just the instruction text, no explanation.`
      .trim();

    userPrompt = `
generate a new, improved system instruction that will achieve higher accuracy than previous attempts. 

task: 
${task[stageName]}

past prompt performance:
${performanceAnalysis}

output examples:
${dataSummary}

cosine similarity score to beat: 
${bestScore.toFixed(3)}`.trim();

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
    traceLog("new prompt generated", { stage: stageName, instruction: newInstruction });

    return {
      instruction: newInstruction,
      examples: [],
    };
  };
}
