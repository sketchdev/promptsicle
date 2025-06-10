import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import { Evaluator, MIPROv2, Pipeline, Prompt, Proposer } from "../mipro.ts";

type RunOutput = { predicted: string; target: string; explanation: string };
type Item = { text: string; target: string };

/** utility function for consistent trace logging */
function traceLog(message: string, data?: unknown) {
  const timestamp = new Date().toISOString().slice(11, 23);
  console.log(`[${timestamp}] ${message}`);
  if (data !== undefined) {
    console.log(`  └─ ${JSON.stringify(data, null, 2)}`);
  }
}

const data: Item[] = [
  { text: "I loved this movie, it was fantastic", target: "positive" },
  { text: "The plot was boring and slow", target: "negative" },
  { text: "It was just okay, nothing special", target: "neutral" },
  { text: "Absolutely wonderful experience", target: "positive" },
  { text: "Terrible acting and awful script", target: "negative" },
  { text: "The new update changed everything in a way I never expected.", target: "negative" },
  { text: "I can't believe bad guy won!", target: "negative" },
  { text: "Well, that meeting was interesting, to say the least.", target: "negative" },
  { text: "He smiled after reading the report, then quietly walked out without a word.", target: "negative" },
  { text: "Your cooking tonight was truly something special.", target: "positive" },
];

const openai = new OpenAI();

/** Pipeline with two sequential LLM stages: label then explain. */
class SentimentPipeline implements Pipeline<RunOutput> {
  stageNames() {
    return ["label", "explain"];
  }

  async run(item: Item, prompts: Record<string, Prompt>) {
    traceLog(`pipeline running for item: "${item.text.slice(0, 50)}..."`);

    const { instruction, examples } = prompts.label;

    const labelMessages: OpenAI.Responses.EasyInputMessage[] = [
      ...examples.flatMap((e) => [
        { role: "user" as const, content: e.input },
        { role: "assistant" as const, content: e.output },
      ]),
      { role: "user" as const, content: item.text },
    ];

    traceLog("calling label stage", { instruction: instruction.slice(0, 100) + "..." });
    const labelResp = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions: instruction,
      input: labelMessages,
      temperature: 0,
    });

    const predicted = labelResp.output_text.trim().toLowerCase();
    traceLog(`label stage result: ${predicted} (target: ${item.target})`);

    const explainPrompt = prompts.explain.instruction;
    traceLog("calling explain stage", { instruction: explainPrompt.slice(0, 100) + "..." });
    const expResp = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions: explainPrompt,
      input: [
        { role: "user" as const, content: `Text: "${item.text}"\nLabel: ${predicted}` },
      ],
      temperature: 0.7,
    });

    const result = { predicted, target: item.target, explanation: expResp.output_text.trim() };
    traceLog("pipeline complete", { predicted, target: item.target, explanationLength: result.explanation.length });
    return result;
  }
}

const proposer: Proposer = async ({ stageName, pastAttempts }) => {
  traceLog(`proposer called for stage: ${stageName}`, { pastAttemptsCount: pastAttempts.length });

  const basePrompts = {
    label: "Classify the sentiment of the text as positive, negative, or neutral. Respond with the single word label.",
    explain: "Explain in one sentence why the given label fits the text.",
  };

  let systemPrompt = "";
  let userPrompt = "";

  if (pastAttempts.length === 0) {
    traceLog("using base prompt for first iteration");
    // first iteration - just return the base prompt
    return {
      instruction: basePrompts[stageName as keyof typeof basePrompts],
      examples: stageName === "label"
        ? [
          { input: "I adore this song", output: "positive" },
          { input: "This meal tastes awful", output: "negative" },
          { input: "It is an average day", output: "neutral" },
        ]
        : [],
    };
  }

  // analyze past attempts and generate improvements
  const performanceAnalysis = pastAttempts
    .map((attempt, i) => `attempt ${i + 1}: "${attempt.prompt.instruction}" (score: ${attempt.score.toFixed(3)})`)
    .join("\n");

  const bestScore = Math.max(...pastAttempts.map((a) => a.score));
  const worstScore = Math.min(...pastAttempts.map((a) => a.score));

  traceLog("analyzing past attempts", { bestScore, worstScore, attemptCount: pastAttempts.length });

  if (stageName === "label") {
    systemPrompt =
      `you are an expert prompt engineer specializing in text classification tasks. your goal is to improve prompts for sentiment classification that achieve higher accuracy.

key requirements:
- the output must be exactly one word: "positive", "negative", or "neutral"
- the prompt should be clear and unambiguous
- consider edge cases and nuanced language
- maintain consistency across different text types`;

    userPrompt = `task: sentiment classification (positive/negative/neutral)

past prompt performance:
${performanceAnalysis}

best score so far: ${bestScore.toFixed(3)}
worst score so far: ${worstScore.toFixed(3)}

generate a new, improved system instruction that will achieve higher accuracy than previous attempts. focus on clarity, specificity, and handling edge cases. respond with just the instruction text, no explanation.`;
  } else {
    systemPrompt =
      `you are an expert prompt engineer specializing in explanation generation. your goal is to improve prompts that generate clear, concise explanations for sentiment classifications.

key requirements:
- explanations should be exactly one sentence
- should clearly connect the text content to the predicted label
- should be informative and specific
- should help users understand the reasoning`;

    userPrompt = `task: generate explanations for sentiment classification results

past prompt performance:
${performanceAnalysis}

best score so far: ${bestScore.toFixed(3)}
worst score so far: ${worstScore.toFixed(3)}

generate a new, improved system instruction for creating explanations. the explanations should be more helpful and accurate than previous attempts. respond with just the instruction text, no explanation.`;
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
    examples: stageName === "label"
      ? [
        { input: "I adore this song", output: "positive" },
        { input: "This meal tastes awful", output: "negative" },
        { input: "It is an average day", output: "neutral" },
      ]
      : [],
  };
};

/** Accuracy‑style evaluator. */
const evaluator: Evaluator<RunOutput> = (outs: RunOutput[]) => {
  const correct = outs.filter((o) => o.predicted === o.target).length;
  const accuracy = correct / outs.length;
  traceLog(`evaluation complete: ${correct}/${outs.length} correct (${(accuracy * 100).toFixed(1)}%)`);
  return accuracy;
};

export default async function sentiment() {
  traceLog("starting miprov2 optimization", { maxIterations: 20, batchSize: 3, dataSize: data.length });

  const mipro = new MIPROv2(
    new SentimentPipeline(),
    proposer,
    evaluator,
    data,
    { maxIterations: 20, batchSize: 3 },
  );

  const initialPrompts = {
    label: {
      instruction: "Classify the sentiment of the text.",
      examples: [],
    },
    explain: {
      instruction: "Explain the sentiment in one concise sentence.",
      examples: [],
    },
  };

  traceLog("starting compilation with initial prompts", initialPrompts);

  const best = await mipro.optimize(initialPrompts);

  traceLog("optimization complete - best prompts found:");
  console.log("Best label instruction:", best.label.instruction);
  console.log("Best explain instruction:", best.explain.instruction);
}
