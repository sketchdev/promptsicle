import { DataLoader, Item } from "@/types.ts";
import * as path from "jsr:@std/path";
import { parse } from "jsr:@std/yaml";

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
