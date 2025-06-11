import { Prompt } from "@/types.ts";

/** utility function for consistent trace logging */
export function traceLog(message: string, data?: unknown) {
  const timestamp = new Date().toISOString().slice(11, 23);
  console.log(`[${timestamp}] ${message}`);
  if (data !== undefined) {
    console.log(`  └─ ${JSON.stringify(data, null, 2)}`);
  }
}

export function promptBuilder(
  instruction: string,
  examples?: Record<string, unknown>[],
): Prompt {
  return {
    instruction,
    examples: examples
      ? examples.map((e) => ({
        input: e.input as string,
        output: e.output as string,
      }))
      : [],
  };
}

export function singleStagePromptBuilder(
  instruction: string,
  examples?: Record<string, unknown>[],
): Record<"generate", Prompt> {
  return {
    generate: promptBuilder(instruction, examples),
  };
}
