import { semanticEvaluator } from "@/evaluators.ts";
import { yamlFileLoader } from "@/loaders.ts";
import { MIPROv2 } from "@/mipro.ts";
import { singleStagePromptBuilder } from "@/pipelines.ts";
import { llmProposer } from "@/proposers.ts";
import { consoleOutputter } from "../../output.ts";
import { singleStageRunner } from "../../runners.ts";

export default async function recipe() {
  const mipro = new MIPROv2(
    ["generate"],
    singleStageRunner(),
    yamlFileLoader("./examples/recipe/data"),
    llmProposer("recipe generation"),
    semanticEvaluator,
    singleStagePromptBuilder("Create a recipe for the dish provided"),
    consoleOutputter,
    { maxIterations: 5, batchSize: 3 },
  );

  await mipro.optimize();
}
