import {
  MessageCreateParamsBase,
  MessageParam,
  Tool,
} from "@anthropic-ai/sdk/resources/messages";
import { get_encoding } from "tiktoken";
import { sessionUsageCache, Usage } from "./cache";
import { readFile } from 'fs/promises'

const enc = get_encoding("cl100k_base");

const calculateTokenCount = (
  messages: MessageParam[],
  system: any,
  tools: Tool[]
) => {
  let tokenCount = 0;
  if (Array.isArray(messages)) {
    messages.forEach((message) => {
      if (typeof message.content === "string") {
        tokenCount += enc.encode(message.content).length;
      } else if (Array.isArray(message.content)) {
        message.content.forEach((contentPart: any) => {
          if (contentPart.type === "text") {
            tokenCount += enc.encode(contentPart.text).length;
          } else if (contentPart.type === "tool_use") {
            tokenCount += enc.encode(JSON.stringify(contentPart.input)).length;
          } else if (contentPart.type === "tool_result") {
            tokenCount += enc.encode(
              typeof contentPart.content === "string"
                ? contentPart.content
                : JSON.stringify(contentPart.content)
            ).length;
          }
        });
      }
    });
  }
  if (typeof system === "string") {
    tokenCount += enc.encode(system).length;
  } else if (Array.isArray(system)) {
    system.forEach((item: any) => {
      if (item.type !== "text") return;
      if (typeof item.text === "string") {
        tokenCount += enc.encode(item.text).length;
      } else if (Array.isArray(item.text)) {
        item.text.forEach((textPart: any) => {
          tokenCount += enc.encode(textPart || "").length;
        });
      }
    });
  }
  if (tools) {
    tools.forEach((tool: Tool) => {
      if (tool.description) {
        tokenCount += enc.encode(tool.name + tool.description).length;
      }
      if (tool.input_schema) {
        tokenCount += enc.encode(JSON.stringify(tool.input_schema)).length;
      }
    });
  }
  return tokenCount;
};

interface ProviderTarget {
  name: string;
  references: any[];
  primary: any;
}

const collectProviderTargets = (config: any): ProviderTarget[] => {
  const map = new Map<string, ProviderTarget>();

  const registerProviders = (providers?: any[]) => {
    if (!Array.isArray(providers)) {
      return;
    }

    providers.forEach((provider: any) => {
      if (!provider?.name) {
        return;
      }

      const existing = map.get(provider.name);
      if (existing) {
        if (!existing.references.includes(provider)) {
          existing.references.push(provider);
        }
      } else {
        map.set(provider.name, {
          name: provider.name,
          references: [provider],
          primary: provider,
        });
      }
    });
  };

  registerProviders(config?.Providers);
  registerProviders(config?.providers);

  return Array.from(map.values());
};

const getUseModel = async (
  req: any,
  tokenCount: number,
  config: any,
  lastUsage?: Usage | undefined
) => {
  if (req.body.model.includes(",")) {
    const [provider, model] = req.body.model.split(",");
    const finalProvider = config.Providers.find(
        (p: any) => p.name.toLowerCase() === provider
    );
    const finalModel = finalProvider?.models?.find(
        (m: any) => m.toLowerCase() === model
    );
    if (finalProvider && finalModel) {
      return `${finalProvider.name},${finalModel}`;
    }
    return req.body.model;
  }

  // if tokenCount is greater than the configured threshold, use the long context model
  const longContextThreshold = config.Router.longContextThreshold || 60000;
  const lastUsageThreshold =
    lastUsage &&
    lastUsage.input_tokens > longContextThreshold &&
    tokenCount > 20000;
  const tokenCountThreshold = tokenCount > longContextThreshold;
  if (
    (lastUsageThreshold || tokenCountThreshold) &&
    config.Router.longContext
  ) {
        req.log.info(
      `Using long context model due to token count: ${tokenCount}, threshold: ${longContextThreshold}`
    );
    return config.Router.longContext;
  }
  if (
    req.body?.system?.length > 1 &&
    req.body?.system[1]?.text?.startsWith("<CCR-SUBAGENT-MODEL>")
  ) {
    const model = req.body?.system[1].text.match(
      /<CCR-SUBAGENT-MODEL>(.*?)<\/CCR-SUBAGENT-MODEL>/s
    );
    if (model) {
      req.body.system[1].text = req.body.system[1].text.replace(
        `<CCR-SUBAGENT-MODEL>${model[1]}</CCR-SUBAGENT-MODEL>`,
        ""
      );
      return model[1];
    }
  }
  // If the model is claude-3-5-haiku, use the background model
  if (
    req.body.model?.startsWith("claude-3-5-haiku") &&
    config.Router.background
  ) {
    req.log.info(`Using background model for ${req.body.model}`);
    return config.Router.background;
  }
  // if exits thinking, use the think model
  if (req.body.thinking && config.Router.think) {
    req.log.info(`Using think model for ${req.body.thinking}`);
    return config.Router.think;
  }
  if (
    Array.isArray(req.body.tools) &&
    req.body.tools.some((tool: any) => tool.type?.startsWith("web_search")) &&
    config.Router.webSearch
  ) {
    return config.Router.webSearch;
  }
  return config.Router!.default;
};

export const router = async (req: any, _res: any, context: any) => {
  const { config, event, server } = context;

  const providerTargets = collectProviderTargets(config);

  if (req.bearerToken && providerTargets.length) {
    const token =
      typeof req.bearerToken === "string" ? req.bearerToken.trim() : "";

    if (token) {
      const requestId = Symbol("dynamic-api-key");
      const overrides: any[] = [];

      providerTargets.forEach((target) => {
        const primary = target.primary;
        if (!primary) {
          return;
        }

        if (typeof primary._static_api_key === "undefined") {
          primary._static_api_key =
            primary.api_key ?? primary.apiKey ?? null;
        }

        const stack: any[] = Array.isArray(primary._dynamic_api_key_stack)
          ? primary._dynamic_api_key_stack
          : (primary._dynamic_api_key_stack = []);

        const activeEntry = stack[stack.length - 1];
        const currentKey =
          activeEntry?.newKey ?? primary.api_key ?? primary.apiKey ?? null;

        if (currentKey === token) {
          return;
        }

        stack.push({ requestId, newKey: token });

        target.references.forEach((ref: any) => {
          ref.api_key = token;
          ref.apiKey = token;
        });

        if (server?.providerService?.getProvider) {
          const existingProvider = server.providerService.getProvider(
            target.name
          );
          if (existingProvider) {
            server.providerService.updateProvider(target.name, {
              apiKey: token,
            });
          }
        }

        overrides.push({
          target,
          requestId,
        });
      });

      if (overrides.length) {
        req.dynamicApiKeyOverrides = overrides;
        console.log(
          `[ROUTER] Applied dynamic API key to ${overrides.length} provider(s)`
        );
      }
    }
  }
  // Parse sessionId from metadata.user_id
  if (req.body.metadata?.user_id) {
    const parts = req.body.metadata.user_id.split("_session_");
    if (parts.length > 1) {
      req.sessionId = parts[1];
    }
  }
  const lastMessageUsage = sessionUsageCache.get(req.sessionId);
  const { messages, system = [], tools }: MessageCreateParamsBase = req.body;
  if (config.REWRITE_SYSTEM_PROMPT && system.length > 1 && system[1]?.text?.includes('<env>')) {
    const prompt = await readFile(config.REWRITE_SYSTEM_PROMPT, 'utf-8');
    system[1].text = `${prompt}<env>${system[1].text.split('<env>').pop()}`
  }

  try {
    const tokenCount = calculateTokenCount(
      messages as MessageParam[],
      system,
      tools as Tool[]
    );

    let model;
    if (config.CUSTOM_ROUTER_PATH) {
      try {
        const customRouter = require(config.CUSTOM_ROUTER_PATH);
        req.tokenCount = tokenCount; // Pass token count to custom router
        model = await customRouter(req, config, {
          event
        });
      } catch (e: any) {
        req.log.error(`failed to load custom router: ${e.message}`);
      }
    }
    if (!model) {
      model = await getUseModel(req, tokenCount, config, lastMessageUsage);
    }
    req.body.model = model;
  } catch (error: any) {
    req.log.error(`Error in router middleware: ${error.message}`);
    req.body.model = config.Router!.default;
  }
  return;
};
