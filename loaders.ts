import { structured } from "@/llm.ts";
import { DataLoader, Item } from "@/types.ts";
import * as path from "jsr:@std/path";
import { parse } from "jsr:@std/yaml";
import mammoth from "npm:mammoth";
import z from "zod";
import { traceLog } from "./utils.ts";

export function yamlFileLoader(dir: string): DataLoader {
  return async (): Promise<Item[]> => {
    const data: Item[] = [];

    for await (const file of Deno.readDir(dir)) {
      if (file.isFile && file.name.endsWith(".yaml")) {
        const filePath = path.join(dir, file.name);
        const content = Deno.readTextFileSync(filePath);
        const item = parse(content) as Item;
        if (item) {
          data.push(item);
        } else {
          throw new Error(`Invalid data format in file ${filePath}`);
        }
      }
    }
    return data;
  };
}

export function docxFileLoader(dir: string): DataLoader {
  return async (): Promise<Item[]> => {
    const data: Item[] = [];

    for await (const file of Deno.readDir(dir)) {
      if (file.isFile && file.name.endsWith(".docx")) {
        traceLog(`Loading file: ${file.name}`);
        const filePath = path.join(dir, file.name);
        const buffer = await Deno.readFile(filePath);
        const convertResult = await mammoth.convertToHtml({ buffer });
        const htmlContent = convertResult.value.replace(/<img[^>]*>/g, ""); // Remove images from HTML content
        const resp = await structured({
          instructions:
            "Summarize the content of this document, and produce a two sentence prompt that a user might type to create the resulting document.",
          input: [{ role: "user" as const, content: htmlContent }],
          format: z.object({ prompt: z.string() }),
          formatName: "summary",
        });
        if (!resp || !resp.prompt) {
          throw new Error(`Failed to summarize document ${filePath}`);
        }
        const item: Item = {
          text: resp.prompt,
          target: htmlContent,
        };
        data.push(item);
        traceLog(`Loaded item from ${file.name}:`, { textLength: item.text.length, targetLength: item.target.length });
      }
    }
    return data;
  };
}
