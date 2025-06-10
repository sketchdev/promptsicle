export interface Configuration {
  trainFolder: string;
  testFolder: string;
}

export type DataLoader = () => Promise<Item[]>;

export type PipelineOutput = { predicted: string; target: string };

export type Item = { text: string; target: string };

export interface Example {
  input: string;
  output: string;
}

export interface Prompt {
  instruction: string;
  examples: Example[];
}

export interface Pipeline<T, TStages extends string = string> {
  /**
   * Execute the pipeline on a single piece of input data with the supplied prompts and return the model output.
   */
  run(input: unknown, prompts: Record<TStages, Prompt>): Promise<T>;

  /**
   * The names of the stages in deterministic order so the optimiser can address them.
   */
  stageNames(): readonly TStages[];
}

/**
 * Evaluator returns a higher-is-better score for a batch of outputs.
 */
export type Evaluator<T> = (outputs: T[]) => Promise<number>;

/**
 * Proposer returns a new Prompt for a given module when supplied with grounding context.
 */
export interface ProposerContext<TStages extends string = string> {
  stageName: TStages;
  dataSummary: string;
  programSummary: string;
  pastAttempts: Array<{ prompt: Prompt; score: number }>;
  initialPrompts: Record<TStages, Prompt>;
}

export type Proposer<TStages extends string = string> = (ctx: ProposerContext<TStages>) => Promise<Prompt>;

export type Outputter<T extends string = string> = (prompts: Record<T, Prompt>) => void;
