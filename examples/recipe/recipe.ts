import { parse } from "jsr:@std/yaml";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import { Evaluator, MIPROv2, Pipeline, Prompt, Proposer } from "../../mipro.ts";

type PipelineOutput = { predicted: string; target: string };
type Item = { text: string; target: string };

/** utility function for consistent trace logging */
function traceLog(message: string, data?: unknown) {
  const timestamp = new Date().toISOString().slice(11, 23);
  console.log(`[${timestamp}] ${message}`);
  if (data !== undefined) {
    console.log(`  └─ ${JSON.stringify(data, null, 2)}`);
  }
}

const openai = new OpenAI();

type RecipeStages = "generate";

class RecipePipeline implements Pipeline<PipelineOutput, RecipeStages> {
  stageNames(): readonly RecipeStages[] {
    return ["generate"] as const;
  }

  async run(item: Item, prompts: Record<RecipeStages, Prompt>): Promise<PipelineOutput> {
    traceLog(`pipeline running for item: "${item.text.slice(0, 50)}..."`);

    const { instruction, examples } = prompts.generate;

    traceLog("calling generate stage", { instruction: instruction.slice(0, 100) + "..." });
    const labelResp = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions: instruction,
      input: [
        ...examples.flatMap((e) => [
          { role: "user" as const, content: e.input },
          { role: "assistant" as const, content: e.output },
        ]),
        { role: "user" as const, content: item.text },
      ],
      temperature: 0,
    });

    const predicted = labelResp.output_text.trim().toLowerCase();
    traceLog(`label stage result: ${predicted} (target: ${item.target})`);

    const result: PipelineOutput = { predicted, target: item.target };
    traceLog("pipeline complete", result);
    return result;
  }
}

const proposer: Proposer<RecipeStages> = async ({ stageName, pastAttempts }): Promise<Prompt> => {
  traceLog(`proposer called for stage: ${stageName}`, { pastAttemptsCount: pastAttempts.length });

  let systemPrompt = "";
  let userPrompt = "";

  if (pastAttempts.length === 0) {
    traceLog("using base prompt for first iteration");
    // first iteration - just return the base prompt
    const baseInstructions: Record<RecipeStages, string> = {
      generate: "Create a recipe for the dish provided",
    };
    return {
      instruction: baseInstructions[stageName as keyof typeof baseInstructions],
      examples: [],
    };
  }

  // analyze past attempts and generate improvements
  const performanceAnalysis = pastAttempts
    .map((attempt, i) => `attempt ${i + 1}: "${attempt.prompt.instruction}" (score: ${attempt.score.toFixed(3)})`)
    .join("\n");

  const bestScore = Math.max(...pastAttempts.map((a) => a.score));
  const worstScore = Math.min(...pastAttempts.map((a) => a.score));

  traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: pastAttempts.length });

  if (stageName === "generate") {
    systemPrompt =
      "you are a professional chef. your goal is to improve prompts for writing recipes based on past performance.";

    userPrompt = `task: recipe generation

past prompt performance:
${performanceAnalysis}

best score so far: ${bestScore.toFixed(3)}
worst score so far: ${worstScore.toFixed(3)}

generate a new, improved system instruction that will achieve higher accuracy than previous attempts. focus on clarity, specificity, and handling edge cases. respond with just the instruction text, no explanation.`;
  }

  traceLog("generating new prompt with llm");
  const responseSchema = z.object({ new_instruction: z.string() });
  const response = await openai.responses.parse({
    model: "gpt-4o-mini",
    instructions: systemPrompt,
    input: [{ role: "user" as const, content: userPrompt }],
    temperature: 0.8,
    text: { format: zodTextFormat(responseSchema, "result") },
  });

  if (!response.output_parsed || !response.output_parsed.new_instruction) {
    throw new Error("Failed to parse response from OpenAI");
  }

  const newInstruction = response.output_parsed.new_instruction;
  traceLog("new prompt generated", { stage: stageName, instructionLength: newInstruction.length });

  return {
    instruction: newInstruction,
    examples: [],
  };
};

const evaluator: Evaluator<PipelineOutput> = (outs: PipelineOutput[]) => {
  const correct = outs.filter((o) => o.predicted === o.target).length;
  const accuracy = correct / outs.length;
  traceLog(`evaluation complete: ${correct}/${outs.length} correct (${(accuracy * 100).toFixed(1)}%)`);
  return accuracy;
};

async function getData(): Promise<Item[]> {
  traceLog("loading recipe data");
  const data: Item[] = [];

  for await (const file of Deno.readDir("./examples/recipe/data")) {
    if (file.isFile && file.name.endsWith(".yaml")) {
      const filePath = `./examples/recipe/data/${file.name}`;
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
}

export default async function recipe() {
  const data = await getData();

  traceLog("starting miprov2 optimization", { maxIterations: 20, batchSize: 3, dataSize: data.length });

  const mipro = new MIPROv2(
    new RecipePipeline(),
    proposer,
    evaluator,
    data,
    { maxIterations: 20, batchSize: 3 },
  );

  const initialPrompts: Record<RecipeStages, Prompt> = {
    generate: {
      instruction: "Create a recipe for the dish provided",
      examples: [],
    },
  };

  traceLog("starting compilation with initial prompts", initialPrompts);

  const best = await mipro.compile(initialPrompts);

  traceLog("optimization complete - best prompt found:");
  console.log("Best instruction:", best.generate.instruction);
}
