import { parseArgs } from "jsr:@std/cli/parse-args";

const args = parseArgs(Deno.args, {
  alias: {
    config: "c",
    help: "h",
  },
  default: {
    config: "promptsicle.json",
  },
});

if (args.help) {
  console.log("Usage: promptsicle [options]");
  console.log("Options:");
  console.log("  -c, --config <file>   Specify the config file (default: promptsicle.json)");
  console.log("  -h, --help            Show this help message");
  Deno.exit(0);
}

console.log(args);
