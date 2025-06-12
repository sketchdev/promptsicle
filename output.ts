import { Prompt } from "@/types.ts";
import { stringify } from "jsr:@std/yaml";

export function consoleOutputter(best: Record<string, Prompt>) {
  Object.entries(best).forEach(([stage, prompt]) => {
    console.log(`Stage: ${stage}`);
    console.log(`${prompt.instruction}\n`);
  });
}

export function jsonOutputter(options: { filePath?: string } = {}) {
  return (best: Record<string, Prompt>) => {
    const jsonContent = JSON.stringify(best, null, 2);
    if (options.filePath) {
      return Deno.writeFile(options.filePath, new TextEncoder().encode(jsonContent));
    } else {
      console.log("Best prompts:");
      console.log(jsonContent);
    }
  };
}

export function yamlOutputter(options: { filePath?: string } = {}) {
  return (best: Record<string, Prompt>) => {
    const yamlContent = stringify(best, { lineWidth: -1 });
    if (options.filePath) {
      return Deno.writeFile(options.filePath, new TextEncoder().encode(yamlContent));
    } else {
      console.log("Best prompts:");
      console.log(yamlContent);
    }
  };
}
