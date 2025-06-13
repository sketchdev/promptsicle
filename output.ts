import { EdgePrompt } from "@/types.ts";

export const edgeStringOutputter = (options: { filePath?: string } = {}) => (bestPrompt: EdgePrompt) => {
  if (options.filePath) {
    return Deno.writeFile(options.filePath, new TextEncoder().encode(bestPrompt.instructions));
  } else {
    console.log("Best prompts:");
    console.log(bestPrompt.instructions);
  }
};
