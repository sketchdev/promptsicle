import { semanticEvaluator } from "@/evaluators.ts";
import { yamlLoader } from "@/loaders.ts";
import { MIPROv2 } from "@/mipro.ts";
import { SingleStagePipeline, singleStagePromptBuilder } from "@/pipelines.ts";
import { llmProposer } from "@/proposers.ts";

export default async function recipe() {
  const mipro = new MIPROv2(
    new SingleStagePipeline(),
    yamlLoader("./examples/recipe/data"),
    llmProposer("recipe generation"),
    semanticEvaluator,
    singleStagePromptBuilder("Create a recipe for the dish provided"),
    { maxIterations: 5, batchSize: 3 },
  );

  const best = await mipro.compile();
  console.log("Best instruction:", best.generate.instruction);
}
