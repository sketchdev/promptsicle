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

const proposer: Proposer<RecipeStages> = async ({ stageName, pastAttempts, dataSummary }): Promise<Prompt> => {
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
    systemPrompt = `your goal is to improve prompts for writing recipes based on past performance.
    
analyze past prompt performance and the provided output examples to try and determine what makes a good recipe prompt.
think about why previous prompts succeeded or failed, and how you can improve clarity, specificity, and handling of edge cases.`;

    userPrompt = `task: recipe generation

past prompt performance:
${performanceAnalysis}

output examples:
${dataSummary}

best score so far: ${bestScore.toFixed(3)}
worst score so far: ${worstScore.toFixed(3)}

generate a new, improved system instruction that will achieve higher accuracy than previous attempts. 
focus on clarity, specificity, and handling edge cases. 
respond with just the instruction text, no explanation.`;
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

const semanticEvaluator: Evaluator<PipelineOutput> = async (outs: PipelineOutput[]) => {
  traceLog("starting semantic evaluation");
  let totalSimilarity = 0;

  for (const out of outs) {
    const [predictedEmbedding, targetEmbedding] = await Promise.all([
      openai.embeddings.create({
        model: "text-embedding-3-small",
        input: out.predicted,
      }),
      openai.embeddings.create({
        model: "text-embedding-3-small",
        input: out.target,
      }),
    ]);

    // calculate cosine similarity
    const pred = predictedEmbedding.data[0].embedding;
    const target = targetEmbedding.data[0].embedding;

    const dotProduct = pred.reduce((sum, val, i) => sum + val * target[i], 0);
    const predMagnitude = Math.sqrt(pred.reduce((sum, val) => sum + val * val, 0));
    const targetMagnitude = Math.sqrt(target.reduce((sum, val) => sum + val * val, 0));

    const similarity = dotProduct / (predMagnitude * targetMagnitude);
    totalSimilarity += similarity;

    traceLog(`semantic similarity: ${similarity.toFixed(3)}`);
  }

  const avgSimilarity = totalSimilarity / outs.length;
  traceLog(`semantic evaluation complete: average similarity ${avgSimilarity.toFixed(3)}`);
  return avgSimilarity;
};

const llmEvaluator: Evaluator<PipelineOutput> = async (outs: PipelineOutput[]) => {
  traceLog("starting llm-based evaluation");
  let totalScore = 0;

  for (const out of outs) {
    const evaluation = await openai.responses.parse({
      model: "gpt-4o-mini",
      instructions: `you are a culinary expert evaluating recipe quality. 
      
      rate the predicted recipe against the target recipe on these criteria:
      - completeness (has ingredients, tools, steps)
      - accuracy (correct cooking methods and ingredients)
      - clarity (easy to follow instructions)
      - overall usefulness
      
      return a score from 0.0 to 1.0 where 1.0 means the predicted recipe is as good as or better than the target.`,
      input: [{
        role: "user" as const,
        content: `target recipe:\n${out.target}\n\npredicted recipe:\n${out.predicted}`,
      }],
      temperature: 0,
      text: {
        format: zodTextFormat(z.object({ score: z.number().min(0).max(1) }), "evaluation"),
      },
    });

    const score = evaluation.output_parsed?.score ?? 0;
    totalScore += score;
    traceLog(`llm evaluation score: ${score.toFixed(3)}`);
  }

  const avgScore = totalScore / outs.length;
  traceLog(`llm evaluation complete: average score ${avgScore.toFixed(3)}`);
  return avgScore;
};

const hybridEvaluator: Evaluator<PipelineOutput> = async (outs: PipelineOutput[]) => {
  traceLog("starting hybrid evaluation");

  // structural completeness check
  const structuralScores = outs.map((out) => {
    const predicted = out.predicted.toLowerCase();
    let score = 0;

    // check for key recipe sections
    if (predicted.includes("ingredient") || predicted.includes("*")) score += 0.25;
    if (predicted.includes("step") || predicted.includes("1.") || predicted.includes("2.")) score += 0.25;
    if (predicted.includes("tool") || predicted.includes("pan") || predicted.includes("bowl")) score += 0.25;
    if (predicted.length > 100) score += 0.25; // reasonable length

    traceLog(`structural score: ${score.toFixed(3)}`);
    return score;
  });

  // semantic similarity (simplified version)
  const semanticScores = await Promise.all(
    outs.map(async (out) => {
      const evaluation = await openai.responses.parse({
        model: "gpt-4o-mini",
        instructions: "rate how semantically similar these two recipes are. return a score from 0.0 to 1.0.",
        input: [{
          role: "user" as const,
          content: `recipe 1:\n${out.target}\n\nrecipe 2:\n${out.predicted}`,
        }],
        temperature: 0,
        text: {
          format: zodTextFormat(z.object({ similarity: z.number().min(0).max(1) }), "similarity"),
        },
      });

      return evaluation.output_parsed?.similarity ?? 0;
    }),
  );

  // combine scores (weighted average)
  const finalScores = structuralScores.map((structural, i) => {
    const semantic = semanticScores[i];
    const combined = (structural * 0.3) + (semantic * 0.7);
    traceLog(
      `combined score: ${combined.toFixed(3)} (structural: ${structural.toFixed(3)}, semantic: ${semantic.toFixed(3)})`,
    );
    return combined;
  });

  const avgScore = finalScores.reduce((sum, score) => sum + score, 0) / finalScores.length;
  traceLog(`hybrid evaluation complete: average score ${avgScore.toFixed(3)}`);
  return avgScore;
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
    semanticEvaluator,
    data,
    { maxIterations: 3, batchSize: 3 },
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
