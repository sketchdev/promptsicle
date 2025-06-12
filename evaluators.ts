import { embedding } from "@/llm.ts";
import { Evaluator, RunnerOutput } from "@/types.ts";
import { traceLog } from "@/utils.ts";

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
