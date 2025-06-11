import { withBackoff } from "@/utils.ts";
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

export function structured<T extends ZodRawShape>(
  params: {
    model?: string;
    instructions: string;
    input: OpenAI.Responses.ResponseInput;
    temperature?: number;
    format: ZodObject<T>;
    formatName: string;
    timeout?: number;
    maxRetries?: number;
  },
) {
  return withBackoff(async () => {
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
    const resp = await openai.responses.parse(req, { timeout: params.timeout ?? 60_000 });
    if (!resp.output_parsed) {
      console.error("Failed to parse response from OpenAI", resp);
      return null;
    }
    return resp.output_parsed as z.infer<typeof params.format>;
  }, params.maxRetries);
}

export function create(params: {
  model?: string;
  instructions: string;
  input: OpenAI.Responses.ResponseInput;
  temperature?: number;
  timeout?: number;
  maxRetries?: number;
}) {
  return withBackoff(async () => {
    const req: OpenAI.Responses.ResponseCreateParamsNonStreaming = {
      model: params.model ?? "gpt-4o-mini",
      instructions: params.instructions,
      input: params.input,
    };
    if (params.temperature !== undefined) {
      req.temperature = params.temperature;
    }
    const resp = await openai.responses.create(req, { timeout: params.timeout ?? 60_000 });
    return resp.output_text;
  }, params.maxRetries);
}
