import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { PiGenAIAttr } from "@oh-my-pi/pi-agent-core";
import * as ai from "@oh-my-pi/pi-ai";
import { Effort } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { generateTaskLabel } from "@oh-my-pi/pi-coding-agent/task/label";
import { generateCommitMessage } from "@oh-my-pi/pi-coding-agent/utils/commit-message-generator";
import { generateSessionTitle } from "@oh-my-pi/pi-coding-agent/utils/title-generator";
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

function getModelOrThrow(id: string) {
	const model = getBundledModel("anthropic", id);
	if (!model) throw new Error(`Expected model ${id}`);
	return model;
}

function createSettings(modelRoles: Record<string, string>) {
	return {
		get(path: string) {
			if (path === "providers.tinyModel") return "online";
			return undefined;
		},
		getModelRole(role: string) {
			return modelRoles[role];
		},
		getStorage() {
			return undefined;
		},
	} as never;
}
function makeTelemetryProbe() {
	const exporter = new InMemorySpanExporter();
	const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
	const onChatUsage = vi.fn();
	return {
		exporter,
		onChatUsage,
		telemetry: { tracer: provider.getTracer("role-thinking-helper-tests"), onChatUsage },
	};
}

function makeCompletion(text: string) {
	return {
		stopReason: "stop",
		content: [{ type: "text", text }],
		usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2 },
	} as never;
}

beforeEach(() => {
	vi.clearAllMocks();
});

afterEach(() => {
	vi.restoreAllMocks();
});

describe("role thinking helper propagation", () => {
	it("passes smol-role thinking to commit message generation", async () => {
		const model = getModelOrThrow("claude-sonnet-4-5");
		const settings = createSettings({
			default: `${model.provider}/${model.id}:high`,
			smol: "@default:minimal",
		});
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeSimpleMock = vi.spyOn(ai, "completeSimple").mockResolvedValue({
			stopReason: "end_turn",
			content: [{ type: "text", text: "fix scope handling" }],
		} as never);

		const message = await generateCommitMessage(`diff --git a/x b/x\n+change\n`, registry as never, settings);
		expect(message).toBe("fix scope handling");
		expect(completeSimpleMock.mock.calls[0]?.[2]).toMatchObject({
			reasoning: Effort.Minimal,
			maxTokens: 1024,
		});
	});

	it("keeps the commit budget reasoning-safe when the catalog disables reasoning", async () => {
		const model = { ...getModelOrThrow("claude-sonnet-4-5"), reasoning: false };
		const settings = createSettings({
			smol: `${model.provider}/${model.id}`,
		});
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeSimpleMock = vi.spyOn(ai, "completeSimple").mockResolvedValue({
			stopReason: "end_turn",
			content: [{ type: "text", text: "fix qwen title budget" }],
		} as never);

		const message = await generateCommitMessage(`diff --git a/x b/x\n+change\n`, registry as never, settings);
		expect(message).toBe("fix qwen title budget");
		expect(completeSimpleMock.mock.calls[0]?.[2]).toMatchObject({
			maxTokens: 1024,
		});
	});

	it("disables reasoning for title generation even when smol role has thinking", async () => {
		const model = getModelOrThrow("claude-sonnet-4-5");
		const settings = createSettings({
			default: `${model.provider}/${model.id}:high`,
			smol: "@default:low",
		});
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeSimpleMock = vi.spyOn(ai, "completeSimple").mockResolvedValue({
			stopReason: "end_turn",
			content: [{ type: "text", text: "<title>Investigate resolver</title>" }],
		} as never);

		const title = await generateSessionTitle("Investigate resolver", registry as never, settings);
		expect(title).toBe("Investigate resolver");
		expect(completeSimpleMock.mock.calls[0]?.[2]).toMatchObject({ disableReasoning: true });
	});
	it("emits one session-title usage event with the stable oneshot kind", async () => {
		const model = getModelOrThrow("claude-sonnet-4-5");
		const settings = createSettings({ smol: `${model.provider}/${model.id}` });
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeImplMock = vi.fn(async () => makeCompletion("<title>Session title</title>"));
		const probe = makeTelemetryProbe();

		const title = await generateSessionTitle(
			"Investigate the title telemetry",
			registry as never,
			settings,
			"session-title",
			undefined,
			undefined,
			undefined,
			undefined,
			{ telemetryConfig: probe.telemetry, completeImpl: completeImplMock as unknown as typeof ai.completeSimple },
		);

		expect(title).toBe("Session title");
		expect(probe.onChatUsage).toHaveBeenCalledTimes(1);
		expect(probe.exporter.getFinishedSpans()[0]?.attributes[PiGenAIAttr.OneshotKind]).toBe("session_title");
	});

	it("emits one task-label usage event with a distinct stable oneshot kind", async () => {
		const model = getModelOrThrow("claude-sonnet-4-5");
		const settings = createSettings({ smol: `${model.provider}/${model.id}` });
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeImplMock = vi.fn(async () => makeCompletion("<title>Task label</title>"));
		const probe = makeTelemetryProbe();

		const label = await generateTaskLabel(
			"Investigate the task label telemetry",
			registry as never,
			settings,
			"task-label",
			undefined,
			{ telemetryConfig: probe.telemetry, completeImpl: completeImplMock as unknown as typeof ai.completeSimple },
		);

		expect(label).toBe("Task label");
		expect(probe.onChatUsage).toHaveBeenCalledTimes(1);
		expect(probe.exporter.getFinishedSpans()[0]?.attributes[PiGenAIAttr.OneshotKind]).toBe("task_label");
	});

	it("emits one isolation commit usage event with the stable oneshot kind", async () => {
		const model = getModelOrThrow("claude-sonnet-4-5");
		const settings = createSettings({ smol: `${model.provider}/${model.id}` });
		const registry = {
			getAvailable: () => [model],
			getApiKey: async () => "test-key",
			resolver: vi.fn(() => async () => "test-key"),
		};
		const completeImplMock = vi.fn(async () => makeCompletion("fix telemetry propagation"));
		const probe = makeTelemetryProbe();

		const message = await generateCommitMessage(
			"diff --git a/x b/x\n+change\n",
			registry as never,
			settings,
			"isolation-commit",
			{ telemetryConfig: probe.telemetry, completeImpl: completeImplMock as unknown as typeof ai.completeSimple },
		);

		expect(message).toBe("fix telemetry propagation");
		expect(probe.onChatUsage).toHaveBeenCalledTimes(1);
		expect(probe.exporter.getFinishedSpans()[0]?.attributes[PiGenAIAttr.OneshotKind]).toBe(
			"isolation_commit_message",
		);
	});
});
