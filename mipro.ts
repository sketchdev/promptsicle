/**
 * MIPROv2 optimizer, a black box prompt tuner for multi stage language model pipelines.
 *
 * The class expects three user supplied items:
 *   • A Pipeline implementation that can execute an end to end program when given an input item and a dictionary of prompts.
 *   • A Proposer that, given grounding context, can generate a new Prompt (instruction plus demonstrations) for a single module.
 *   • An Evaluator that scores a batch of pipeline outputs with a single scalar so the optimiser can compare trials.
 *
 * Credit assignment among modules is handled with a very light Tree Parzen Estimator surrogate. It learns a Gaussian
 * mixture model for good versus bad scores seen so far and uses the ratio of densities to pick the next module to tweak.
 * For simplicity the implementation keeps the surrogate per module independent. The optimiser defaults to random
 * exploration until it has enough samples to fit the model, then it steadily shifts toward exploitation.
 */

import { DataLoader, Evaluator, Item, Outputter, Prompt, Proposer, ProposerContext, Runner } from "@/types.ts";

/* --------------------------------- Surrogate --------------------------------- */

class Surrogate {
  private readonly good: number[] = [];
  private readonly bad: number[] = [];

  /**
   * Update the model with a new score. Threshold is the running median so the sets are balanced.
   */
  update(score: number): void {
    const median = this.median([...this.good, ...this.bad]);
    if (isNaN(median) || score >= median) {
      this.good.push(score);
    } else {
      this.bad.push(score);
    }
  }

  /**
   * Return a utility value, larger values mean more promising.
   */
  utility(score: number): number {
    if (!this.good.length || !this.bad.length) {
      return Math.random();
    }
    const pg = this.parzen(score, this.good);
    const pb = this.parzen(score, this.bad);
    return pg / (pb + 1e-6);
  }

  private parzen(x: number, arr: number[]): number {
    if (!arr.length) {
      return 0;
    }
    const bandwidth = 1e-3 + 1.06 * this.std(arr) * Math.pow(arr.length, -0.2);
    return arr.reduce((sum, v) => sum + this.gaussian(x, v, bandwidth), 0) / arr.length;
  }

  private gaussian(x: number, mu: number, sigma: number): number {
    const coeff = 1 / (Math.sqrt(2 * Math.PI) * sigma);
    const exponent = -((x - mu) ** 2) / (2 * sigma ** 2);
    return coeff * Math.exp(exponent);
  }

  private std(arr: number[]): number {
    const m = this.median(arr);
    const variance = arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
  }

  private median(arr: number[]): number {
    if (!arr.length) {
      return NaN;
    }
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
}

/* --------------------------------- History ---------------------------------- */

interface Trial {
  iteration: number;
  prompts: Record<string, Prompt>;
  score: number;
}

/* -------------------------------- MIPROv2 ----------------------------------- */

export interface MIPROOptions {
  maxIterations?: number;
  batchSize?: number;
  randomSeed?: number;
}

export class MIPROv2<T, TStages extends string = string> {
  private readonly loader: DataLoader;
  private readonly stages: TStages[];
  private readonly runner: Runner<T, TStages>;
  private readonly proposer: Proposer<TStages>;
  private readonly evaluator: Evaluator<T[]>;
  private readonly opts: Required<MIPROOptions>;
  private readonly history: Trial[] = [];
  private readonly surrogates: Record<string, Surrogate> = {};
  private readonly initialPrompts: Record<string, string> | Record<TStages, Prompt>;
  private readonly outputter: Outputter<TStages>;
  private data: Item[] = [];
  private readonly executedStages: Set<TStages> = new Set();

  constructor(
    stages: TStages[],
    runner: Runner<T, TStages>,
    loader: DataLoader,
    proposer: Proposer<TStages>,
    evaluator: Evaluator<T[]>,
    initialPrompts: Record<string, string> | Record<TStages, Prompt> = {} as Record<TStages, Prompt>,
    outputter: Outputter<TStages>,
    opts: MIPROOptions = {},
  ) {
    this.stages = stages;
    this.runner = runner;
    this.loader = loader;
    this.proposer = proposer;
    this.evaluator = evaluator;
    this.initialPrompts = initialPrompts;
    this.outputter = outputter;
    this.opts = {
      maxIterations: opts.maxIterations ?? 100,
      batchSize: opts.batchSize ?? 8,
      randomSeed: opts.randomSeed ?? Date.now(),
    };

    for (const stage of this.stages) {
      this.surrogates[stage] = new Surrogate();
    }
    seedRandom(this.opts.randomSeed);
  }

  /**
   * Optimise prompts for all modules and return the best set discovered.
   */
  async optimize(options: { earlyStopThreshold?: number } = {}): Promise<void> {
    const earlyStopThreshold = options.earlyStopThreshold ?? 0.95;
    this.data = await this.loader();

    const startingPrompts: Record<TStages, Prompt> = {} as Record<TStages, Prompt>;
    for (const initialPrompt of Object.entries(this.initialPrompts)) {
      const [stage, prompt] = initialPrompt;
      if (prompt && typeof prompt === "string") {
        startingPrompts[stage as TStages] = {
          instruction: prompt as string,
          examples: [],
        };
      } else {
        startingPrompts[stage as TStages] = prompt;
      }
    }

    let best: Trial = { iteration: -1, prompts: startingPrompts, score: -Infinity };

    for (let iter = 0; iter < this.opts.maxIterations; iter++) {
      const stage = this.selectStage();
      this.executedStages.add(stage);
      const candidatePrompt = await this.proposePrompt(stage, startingPrompts, best.prompts);
      const candidatePrompts = { ...best.prompts } as Record<TStages, Prompt>;
      candidatePrompts[stage] = candidatePrompt;
      const score = await this.evaluatePrompts(candidatePrompts);

      const trial: Trial = { iteration: iter, prompts: candidatePrompts, score };
      this.history.push(trial);
      this.surrogates[stage].update(score);

      if (score > best.score) {
        best = trial;
        console.log(`MIPROv2 improved to ${score.toFixed(4)} at iteration ${iter}`);

        if (score >= earlyStopThreshold) {
          console.log(`MIPROv2 reached target score of ${earlyStopThreshold}, stopping early at iteration ${iter}`);
          break;
        }
      }
    }

    this.outputter(best.prompts);
  }

  /* ----------------------------- Internals ----------------------------- */

  private selectStage(): TStages {
    // prioritize stages that haven't been executed yet
    const unexecutedStages = this.stages.filter((stage) => !this.executedStages.has(stage));
    if (unexecutedStages.length > 0) {
      const idx = Math.floor(Math.random() * unexecutedStages.length);
      return unexecutedStages[idx];
    }

    // all stages have been executed at least once, use surrogate utilities
    const utils = this.stages.map((stage) => {
      const lastScore = this.history.length ? this.history[this.history.length - 1].score : 0;
      return this.surrogates[stage].utility(lastScore);
    });
    const total = utils.reduce((a, b) => a + b, 0);
    const r = Math.random() * total;
    let acc = 0;
    for (let i = 0; i < utils.length; i++) {
      acc += utils[i];
      if (r <= acc) return this.stages[i];
    }
    const idx = Math.floor(Math.random() * this.stages.length);
    return this.stages[idx];
  }

  private proposePrompt(
    stage: TStages,
    initialPrompts: Record<TStages, Prompt>,
    current: Record<TStages, Prompt>,
  ): Promise<Prompt> {
    void current;

    const past = this.history.map((h) => ({ prompt: h.prompts[stage], score: h.score }));
    const ctx: ProposerContext<TStages> = {
      stageName: stage,
      dataSummary: this.summariseData(),
      programSummary: this.summariseProgram(),
      pastAttempts: past,
      initialPrompts,
    };
    return this.proposer(ctx);
  }

  private async evaluatePrompts(prompts: Record<TStages, Prompt>): Promise<number> {
    const batch = sampleArray(this.data, this.opts.batchSize);
    const outputs = [];
    for (const item of batch) {
      const x = await this.runner(item, prompts);
      outputs.push(x);
    }
    return this.evaluator(outputs);
  }

  private summariseData(): string {
    // Placeholder summary using first few items for quick grounding.
    const preview = JSON.stringify(this.data.slice(0, 3));
    return `Sample data preview: ${preview}`;
  }

  private summariseProgram(): string {
    return `Program stages: ${this.stages.join(", ")}`;
  }
}

/* ------------------------------ Helpers ------------------------------ */

function seedRandom(seed: number) {
  // Mulberry32 PRNG for reproducible runs. Small and sufficient.
  let t = seed + 0x6d2b79f5;
  Math.random = function () {
    t += 0x6d2b79f5;
    let v = Math.imul(t ^ (t >>> 15), 1 | t);
    v = (v + Math.imul(v ^ (v >>> 7), 61 | v)) ^ v;
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
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
