import { embedding, structured } from "@/llm.ts";
import { Evaluator, RunnerOutput } from "@/types.ts";
import { traceLog } from "@/utils.ts";
import z from "zod";

export const semanticEvaluator: Evaluator<RunnerOutput[]> = async (outs: RunnerOutput[]) => {
  traceLog("starting semantic evaluation");
  let totalSimilarity = 0;

  for (const out of outs) {
    const [pred, target] = await Promise.all([
      embedding(out.predicted),
      embedding(out.target),
    ]);

    // calculate cosine similarity
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

const llmEvaluator: Evaluator<RunnerOutput[]> = async (outs: RunnerOutput[]) => {
  traceLog("starting llm-based evaluation");
  let totalScore = 0;

  for (const out of outs) {
    const evaluation = await structured({
      model: "gpt-4o-mini",
      instructions: `you are an expert evaluating content quality. 
      
rate the predicted content against the target content on these criteria:
- completeness (contains all necessary elements)
- accuracy (correct information and methods)
- clarity (easy to understand and follow)
- overall usefulness

return a score from 0.0 to 1.0 where 1.0 means the predicted content is as good as or better than the target.`,
      input: [{
        role: "user" as const,
        content: `target:\n${out.target}\n\npredicted:\n${out.predicted}`,
      }],
      temperature: 0,
      format: z.object({ score: z.number().min(0).max(1) }),
      formatName: "evaluation",
    });

    const score = evaluation?.score ?? 0;
    totalScore += score;
    traceLog(`llm evaluation score: ${score.toFixed(3)}`);
  }

  const avgScore = totalScore / outs.length;
  traceLog(`llm evaluation complete: average score ${avgScore.toFixed(3)}`);
  return avgScore;
};

const hybridRecipeEvaluator: Evaluator<RunnerOutput[]> = async (outs: RunnerOutput[]) => {
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
      const evaluation = await structured({
        model: "gpt-4o-mini",
        instructions: "rate how semantically similar these two recipes are. return a score from 0.0 to 1.0.",
        input: [{
          role: "user" as const,
          content: `recipe 1:\n${out.target}\n\nrecipe 2:\n${out.predicted}`,
        }],
        temperature: 0,
        format: z.object({ similarity: z.number().min(0).max(1) }),
        formatName: "similarity",
      });

      return evaluation?.similarity ?? 0;
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
