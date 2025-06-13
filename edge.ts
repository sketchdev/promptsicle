import {
  DataLoader,
  EdgeEvaluator,
  EdgeHistoryItem,
  EdgeOptimizer,
  EdgeOutputter,
  EdgePrompt,
  EdgeResult,
  EdgeRunner,
  Item,
} from "@/types.ts";

export class Edge {
  private readonly loader: DataLoader;
  private readonly optimizer: EdgeOptimizer;
  private readonly runner: EdgeRunner;
  private readonly evaluator: EdgeEvaluator;
  private readonly outputter: EdgeOutputter;
  private readonly opts: { maxIterations: number; batchSize: number };
  private readonly promptHistory: EdgePrompt[] = [];
  private readonly history: EdgeHistoryItem[] = [];
  private data: Item[] = [];

  constructor(
    loader: DataLoader,
    initialPrompt: string,
    optimizer: EdgeOptimizer,
    runner: EdgeRunner,
    evaluator: EdgeEvaluator,
    outputter: EdgeOutputter,
    opts: { maxIterations?: number; batchSize?: number } = {},
  ) {
    this.loader = loader;
    this.optimizer = optimizer;
    this.runner = runner;
    this.evaluator = evaluator;
    this.outputter = outputter;
    this.opts = {
      maxIterations: opts.maxIterations ?? 20,
      batchSize: opts.batchSize ?? 8,
    };
    this.promptHistory = [{ instructions: initialPrompt, score: -Infinity }];
  }

  async fit(options: { earlyStopThreshold?: number } = {}): Promise<EdgePrompt> {
    const earlyStopThreshold = options.earlyStopThreshold ?? 1;
    const startTime = Date.now();
    this.data = await this.loader();

    console.log("\n");
    console.log(`📊 Dataset: ${this.data.length} items`);
    console.log(`🔄 Max iterations: ${this.opts.maxIterations}`);
    console.log(`📦 Batch size: ${this.opts.batchSize}`);
    console.log(`🏁 Early stop threshold: ${earlyStopThreshold}\n`);

    for (let iter = 0; iter < this.opts.maxIterations; iter++) {
      const bestScoreBefore = this.promptHistory.reduce(
        (best, current) => current.score > best ? current.score : best,
        -Infinity,
      );

      const batch = sampleArray(this.data, this.opts.batchSize);
      const outputs: EdgeResult[] = [];
      for (const item of batch) {
        const x = await this.runner(item, this.promptHistory[this.promptHistory.length - 1]);
        outputs.push({ prompt: this.promptHistory[this.promptHistory.length - 1], predicted: x, target: item.target });
      }
      const score = await this.evaluator(outputs);

      this.promptHistory[this.promptHistory.length - 1].score = score;

      if (score > bestScoreBefore) {
        console.log(`Improved to ${score.toFixed(4)} at iteration ${iter}`);
        if (score >= earlyStopThreshold) {
          console.log(`Reached target score of ${earlyStopThreshold}, stopping early at iteration ${iter}`);
          break;
        }
      }

      const worstOutput = outputs.reduce((worst, current) =>
        current.prompt.score < worst.prompt.score ? current : worst
      );

      this.history.push({
        instructions: worstOutput.prompt.instructions,
        score: worstOutput.prompt.score,
        prediction: worstOutput.predicted,
        target: worstOutput.target,
      });

      // only generate new prompt if we have more iterations to go
      if (iter < this.opts.maxIterations - 1) {
        const newPrompt = await this.optimizer(this.history);
        this.promptHistory.push({ instructions: newPrompt, score: -Infinity });
      }
    }

    const endTime = Date.now();
    const totalTime = (endTime - startTime) / 1000;
    console.log(`\n⏱️  Total optimization time: ${totalTime.toFixed(2)}s`);

    const bestPrompt = this.promptHistory.reduce((best, current) => current.score > best.score ? current : best);
    this.outputter(bestPrompt);
    return bestPrompt;
  }
}

function sampleArray<T>(arr: T[], n: number): T[] {
  const copy = [...arr];
  const out: T[] = [];
  while (out.length < n && copy.length) {
    const idx = Math.floor(Math.random() * copy.length);
    out.push(copy.splice(idx, 1)[0]);
  }
  return out;
}
