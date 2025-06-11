import { Prompt, Proposer } from "@/types.ts";
import { traceLog } from "@/utils.ts";
import z from "zod";
import { structured } from "./llm.ts";

export function llmProposer<TStages extends string = string>(task: Record<TStages, string>): Proposer<TStages> {
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

    systemPrompt = `your goal is to improve prompts for the given task (${task[stageName]}) based on past performance.
    
analyze past prompt performance and the provided output examples to try and determine what makes a good prompt for this task.
think about why previous prompts succeeded or failed, and how you can improve clarity, specificity, and handling of edge cases.`;

    userPrompt = `task: ${task[stageName]}

past prompt performance:
${performanceAnalysis}

output examples:
${dataSummary}

best score so far: ${bestScore.toFixed(3)}
worst score so far: ${worstScore.toFixed(3)}

generate a new, improved system instruction that will achieve higher accuracy than previous attempts. 
focus on clarity, specificity, and handling edge cases. 
respond with just the instruction text, no explanation.`;

    const response = await structured({
      model: "gpt-4o-mini",
      instructions: systemPrompt,
      input: [{ role: "user" as const, content: userPrompt }],
      temperature: 0.8,
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
