import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z, ZodObject, ZodRawShape } from "zod";

const openai = new OpenAI();

export async function embedding(input: string) {
  const resp = await openai.embeddings.create({
    input,
    model: "text-embedding-3-small",
  });
  return resp.data[0].embedding;
}

export async function structured<T extends ZodRawShape>(
  params: {
    model?: string;
    instructions: string;
    input: OpenAI.Responses.ResponseInput;
    temperature?: number;
    format: ZodObject<T>;
    formatName: string;
  },
) {
  const req: OpenAI.Responses.ResponseCreateParamsNonStreaming = {
    model: params.model ?? "gpt-4o-mini",
    instructions: params.instructions,
    input: params.input,
    text: {
      format: zodTextFormat(params.format, params.formatName),
    },
  };
  if (params.temperature !== undefined) {
    req.temperature = params.temperature;
  }
  const resp = await openai.responses.parse(req);
  if (!resp.output_parsed) {
    console.error("Failed to parse response from OpenAI", resp);
    return null;
  }
  return resp.output_parsed as z.infer<typeof params.format>;
}

export async function create(params: {
  model?: string;
  instructions: string;
  input: OpenAI.Responses.ResponseInput;
  temperature?: number;
}) {
  const req: OpenAI.Responses.ResponseCreateParamsNonStreaming = {
    model: "gpt-4o-mini",
    instructions: params.instructions,
    input: params.input,
  };
  if (params.temperature !== undefined) {
    req.temperature = params.temperature;
  }
  const resp = await openai.responses.create({
    model: "gpt-4o-mini",
    instructions: params.instructions,
    input: params.input,
  });
  return resp.output_text;
}
