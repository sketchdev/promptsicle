import { embedding } from "@/llm.ts";
import { EdgeResult } from "@/types.ts";
import { traceLog } from "@/utils.ts";

export const edgeEvaluator = () => async (results: EdgeResult[]): Promise<number> => {
  traceLog("starting semantic evaluation");
  let totalSimilarity = 0;

  for (const result of results) {
    const [pred, target] = await Promise.all([
      embedding(result.predicted),
      embedding(result.target),
    ]);

    // calculate cosine similarity
    const dotProduct = pred.reduce((sum, val, i) => sum + val * target[i], 0);
    const predMagnitude = Math.sqrt(pred.reduce((sum, val) => sum + val * val, 0));
    const targetMagnitude = Math.sqrt(target.reduce((sum, val) => sum + val * val, 0));

    const similarity = dotProduct / (predMagnitude * targetMagnitude);
    totalSimilarity += similarity;

    traceLog(`semantic similarity: ${similarity.toFixed(3)}`);
  }

  const avgSimilarity = totalSimilarity / results.length;
  traceLog(`semantic evaluation complete: average similarity ${avgSimilarity.toFixed(3)}`);
  return avgSimilarity;
};
