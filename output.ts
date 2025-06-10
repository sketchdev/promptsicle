import { Prompt } from "@/types.ts";

export function consoleOutputter(best: Record<string, Prompt>) {
  Object.entries(best).forEach(([stage, prompt]) => {
    console.log(`Stage: ${stage}`);
    console.log(`${prompt.instruction}\n`);
  });
}

export function jsonOutputter(best: Record<string, Prompt>) {
  console.log(JSON.stringify(best, null, 2));
}