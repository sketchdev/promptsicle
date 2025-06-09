import { parseArgs } from "jsr:@std/cli/parse-args";
import sentiment from "./examples/sentiment.ts";

const args = parseArgs(Deno.args, {
  alias: {
    configFile: "c",
    help: "h",
  },
});

if (args.help) {
  console.log("Usage: promptsicle [options]");
  console.log("Options:");
  console.log("  -c, --config <file>   Specify the config file (default: promptsicle.json)");
  console.log("  -h, --help            Show this help message");
  Deno.exit(0);
}

await sentiment();
// await run(config.trainFolder, config.testFolder);
console.log("promptsicle completed successfully.");
