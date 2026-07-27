import { afterEach, describe, expect, it, vi } from "bun:test";
import { PiGenAIAttr } from "@oh-my-pi/pi-agent-core";
import type * as ai from "@oh-my-pi/pi-ai";
import type { AssistantMessage } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import {
	classifyUnexpectedStop,
	isUnexpectedStopCandidate,
	parseUnexpectedStopClassification,
} from "@oh-my-pi/pi-coding-agent/session/unexpected-stop-classifier";
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

function makeTelemetryProbe() {
	const exporter = new InMemorySpanExporter();
	const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
	const onChatUsage = vi.fn();
	return {
		exporter,
		onChatUsage,
		telemetry: { tracer: provider.getTracer("unexpected-stop-classifier-tests"), onChatUsage },
	};
}

function makeCompletion(text: string) {
	return {
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		stopReason: "stop",
		content: [{ type: "text", text }],
		usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2 },
	} as never;
}

function makeAssistantMessage(options: {
	stopReason: AssistantMessage["stopReason"];
	content: AssistantMessage["content"];
}): AssistantMessage {
	return {
		role: "assistant",
		provider: "mock",
		model: "mock/mock",
		api: "mock" as unknown as AssistantMessage["api"],
		content: options.content,
		stopReason: options.stopReason,
		timestamp: Date.now(),
	} as unknown as AssistantMessage;
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe("isUnexpectedStopCandidate", () => {
	it("returns true for a text-only stop", () => {
		const message = makeAssistantMessage({
			stopReason: "stop",
			content: [{ type: "text", text: "I should do the same for the JS eval worker." }],
		});
		expect(isUnexpectedStopCandidate(message)).toBe(true);
	});

	it("returns false when stopReason is not stop", () => {
		const length = makeAssistantMessage({
			stopReason: "length",
			content: [{ type: "text", text: "I should continue." }],
		});
		expect(isUnexpectedStopCandidate(length)).toBe(false);

		const aborted = makeAssistantMessage({
			stopReason: "aborted",
			content: [{ type: "text", text: "I should continue." }],
		});
		expect(isUnexpectedStopCandidate(aborted)).toBe(false);
	});

	it("returns false when the message contains a toolCall", () => {
		const message = makeAssistantMessage({
			stopReason: "stop",
			content: [
				{ type: "text", text: "I will run the tests now." },
				{ type: "toolCall", id: "call-1", name: "bash", arguments: {} },
			],
		});
		expect(isUnexpectedStopCandidate(message)).toBe(false);
	});

	it("returns false when the text is only whitespace", () => {
		const message = makeAssistantMessage({
			stopReason: "stop",
			content: [{ type: "text", text: "   \n\t  " }],
		});
		expect(isUnexpectedStopCandidate(message)).toBe(false);
	});

	it("returns false for an empty stop", () => {
		const message = makeAssistantMessage({
			stopReason: "stop",
			content: [],
		});
		expect(isUnexpectedStopCandidate(message)).toBe(false);
	});
});

describe("classifyUnexpectedStop", () => {
	it("uses a reasoning-safe online classifier budget when the catalog disables reasoning", async () => {
		const baseModel = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!baseModel) throw new Error("Expected bundled Claude Sonnet 4.5 model");
		const model = { ...baseModel, reasoning: false };
		const settings = {
			get(path: string) {
				if (path === "providers.unexpectedStopModel") return "online";
				return undefined;
			},
			getModelRole(role: string) {
				return role === "smol" ? `${model.provider}/${model.id}` : undefined;
			},
			getStorage() {
				return undefined;
			},
		} as never;
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: () => async () => "test-key",
		} as never;
		const completeSimpleMock = vi.fn(async (..._args: Parameters<typeof ai.completeSimple>) => makeCompletion("YES"));
		const probe = makeTelemetryProbe();

		const result = await classifyUnexpectedStop("I will continue with the next command.", {
			settings,
			registry,
			sessionId: "session-1",
			telemetryConfig: probe.telemetry,
			completeImpl: completeSimpleMock as unknown as typeof ai.completeSimple,
		});
		const options = completeSimpleMock.mock.calls[0]?.[2] as
			| { disableReasoning?: boolean; maxTokens?: number }
			| undefined;

		expect(result).toBe(true);
		expect(options).toMatchObject({ disableReasoning: true, maxTokens: 1024 });
		const spans = probe.exporter.getFinishedSpans();
		expect(probe.onChatUsage).toHaveBeenCalledTimes(1);
		expect(spans).toHaveLength(1);
		expect(spans[0]?.attributes[PiGenAIAttr.OneshotKind]).toBe("unexpected_stop_classifier");
	});
});

describe("parseUnexpectedStopClassification", () => {
	it("returns true for YES output", () => {
		expect(parseUnexpectedStopClassification("YES")).toBe(true);
		expect(parseUnexpectedStopClassification("yes")).toBe(true);
		expect(parseUnexpectedStopClassification("  Yes, this is unexpected  ")).toBe(true);
	});

	it("returns false for NO output", () => {
		expect(parseUnexpectedStopClassification("NO")).toBe(false);
		expect(parseUnexpectedStopClassification("no")).toBe(false);
		expect(parseUnexpectedStopClassification("No, the task is complete.")).toBe(false);
	});

	it("returns undefined for unparseable output", () => {
		expect(parseUnexpectedStopClassification("maybe")).toBeUndefined();
		expect(parseUnexpectedStopClassification("")).toBeUndefined();
		expect(parseUnexpectedStopClassification("I don't know")).toBeUndefined();
	});
});
