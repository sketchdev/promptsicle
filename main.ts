import { parseArgs } from "jsr:@std/cli/parse-args";
import { defaultConfig } from "./config.ts";
import { run } from "./runner.ts";

const args = parseArgs(Deno.args, {
  alias: {
    configFile: "c",
    help: "h",
  },
});

if (args.help) {
  console.log("Usage: promptsicle [options]");
  console.log("Options:");
  console.log(
    "  -c, --config <file>   Specify the config file (default: promptsicle.json)",
  );
  console.log("  -h, --help            Show this help message");
  Deno.exit(0);
}

let config = defaultConfig;

if (args.configFile) {
  try {
    const configExists = await Deno.stat(args.configFile);
    if (!configExists.isFile) {
      console.error(
        `Config file \`${args.config}\` does not exist or is not a file.`,
      );
      Deno.exit(1);
    }
    config = JSON.parse(await Deno.readTextFile(args.configFile));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Error: ${message}`);
    Deno.exit(1);
  }
}

await run(config.trainFolder, config.testFolder);
console.log("promptsicle completed successfully.");
