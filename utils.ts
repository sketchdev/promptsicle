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

export async function withBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000,
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries) {
        throw lastError;
      }

      traceLog("An error occurred, retrying with backoff", {
        attempt,
        error: error instanceof Error ? error.message : String(error),
      });

      // exponential backoff with jitter
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}
