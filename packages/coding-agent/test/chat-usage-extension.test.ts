import { afterEach, describe, expect, it, vi } from "bun:test";
import type { ChatUsageEvent } from "@oh-my-pi/pi-agent-core";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { createAgentSession, type ExtensionFactory } from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { logger, TempDir } from "@oh-my-pi/pi-utils";

type Harness = {
	tempDir: TempDir;
	authStorage: AuthStorage;
	session: AgentSession;
};

const workspaceTree = (cwd: string) => ({
	rootPath: cwd,
	rendered: "",
	truncated: false,
	totalLines: 0,
	agentsMdFiles: [],
});

function usage(id: string, agentId = "main"): ChatUsageEvent {
	return {
		eventId: crypto.randomUUID(),
		span: undefined as never,
		agent: { id: agentId, name: agentId },
		conversationId: id,
		stepNumber: 0,
		model: "test-model",
		provider: "test-provider",
		modelProvider: "test-provider",
		serviceTier: undefined,
		usage: {
			inputTokens: 10,
			outputTokens: 5,
			totalTokens: 15,
			cachedInputTokens: 2,
			cacheWriteTokens: 1,
			reasoningOutputTokens: 3,
		},
		cost: undefined,
		attributes: undefined,
		headers: undefined,
	};
}

async function createHarness(options: {
	onUsage?: (event: ChatUsageEvent) => void | Promise<void>;
	received: string[];
	onReceived?: (conversationId: string) => void | Promise<void>;
}): Promise<Harness> {
	const tempDir = TempDir.createSync("@pi-chat-usage-");
	const cwd = tempDir.path();
	const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
	authStorage.setRuntimeApiKey("openai", "test-key");
	const modelRegistry = new ModelRegistry(authStorage, tempDir.join("models.yml"));
	const sessionManager = SessionManager.inMemory(cwd);
	const extension: ExtensionFactory = pi => {
		pi.on("chat_usage", async event => {
			const conversationId = event.usage.conversationId ?? "missing";
			options.received.push(conversationId);
			await options.onReceived?.(conversationId);
		});
	};
	const result = await createAgentSession({
		cwd,
		agentDir: tempDir.path(),
		authStorage,
		modelRegistry,
		sessionManager,
		settings: Settings.isolated({ "async.enabled": false }),
		model: getBundledModel("openai", "gpt-4o-mini"),
		telemetry: options.onUsage ? { onChatUsage: options.onUsage } : undefined,
		extensions: [extension],
		disableExtensionDiscovery: true,
		skills: [],
		contextFiles: [],
		workspaceTree: workspaceTree(cwd),
		promptTemplates: [],
		slashCommands: [],
		enableMCP: false,
		enableLsp: false,
	});
	return { tempDir, authStorage, session: result.session };
}

const harnesses: Harness[] = [];

afterEach(async () => {
	for (const harness of harnesses.splice(0)) {
		await harness.session.dispose();
		harness.authStorage.close();
		harness.tempDir.removeSync();
	}
});

describe("chat_usage extension bridge", () => {
	it("delivers one event per primary, inherited child, advisor, and oneshot callback", async () => {
		const received: string[] = [];
		const callbackOrder: string[] = [];
		const { promise: primaryDelivered, resolve: resolvePrimary } = Promise.withResolvers<void>();
		const previous = vi.fn((event: ChatUsageEvent) => {
			callbackOrder.push(`previous:${event.conversationId}`);
		});
		const parent = await createHarness({
			received,
			onUsage: event => previous(event),
			onReceived: conversationId => {
				if (conversationId === "primary") resolvePrimary();
			},
		});
		harnesses.push(parent);
		const parentTelemetry = parent.session.agent.telemetry;
		if (!parentTelemetry?.onChatUsage) throw new Error("Expected SDK to install chat usage telemetry");

		// The callback can fire before the normal mode initializes extensions. The
		// runner must retain this record and replay it exactly once at initialize.
		await parentTelemetry.onChatUsage(usage("primary"));
		await initializeExtensions(parent.session, {
			reportSendError: () => {},
			reportRuntimeError: () => {},
		});
		await primaryDelivered;
		await parentTelemetry.onChatUsage(usage("advisor", "advisor"));
		await parentTelemetry.onChatUsage(usage("oneshot", "oneshot"));

		const child = await createHarness({
			received,
			onUsage: parentTelemetry.onChatUsage,
		});
		harnesses.push(child);
		// Child creation receives the parent's already-bridged callback. It must
		// preserve callback identity rather than wrapping a second extension hop.
		expect(child.session.agent.telemetry?.onChatUsage).toBe(parentTelemetry.onChatUsage);
		await child.session.agent.telemetry?.onChatUsage?.(usage("subagent", "subagent"));

		expect(received).toEqual(["primary", "advisor", "oneshot", "subagent"]);
		expect(previous).toHaveBeenCalledTimes(4);
		expect(callbackOrder).toEqual(["previous:primary", "previous:advisor", "previous:oneshot", "previous:subagent"]);
	});

	it("delivers extension usage in invocation order before a slow previous callback settles", async () => {
		const received: string[] = [];
		const previousGate = Promise.withResolvers<void>();
		const extensionDelivered = Promise.withResolvers<void>();
		const previous = vi.fn(async (event: ChatUsageEvent) => {
			if (event.conversationId === "slow") await previousGate.promise;
		});
		const harness = await createHarness({
			received,
			onUsage: previous,
			onReceived: conversationId => {
				if (conversationId === "fast") extensionDelivered.resolve();
			},
		});
		harnesses.push(harness);
		await initializeExtensions(harness.session, {
			reportSendError: () => {},
			reportRuntimeError: () => {},
		});
		const callback = harness.session.agent.telemetry?.onChatUsage;
		if (!callback) throw new Error("Expected SDK to install chat usage telemetry");

		const slow = callback(usage("slow"));
		const fast = callback(usage("fast"));
		await extensionDelivered.promise;
		expect(received).toEqual(["slow", "fast"]);
		expect(previous).toHaveBeenCalledTimes(2);
		previousGate.resolve();
		await Promise.all([slow, fast]);
	});

	it("replays buffered usage through one FIFO during initialization", async () => {
		const received: string[] = [];
		const firstHandlerStarted = Promise.withResolvers<void>();
		const firstHandlerGate = Promise.withResolvers<void>();
		const secondHandlerDelivered = Promise.withResolvers<void>();
		const harness = await createHarness({
			received,
			onUsage: () => {},
			onReceived: async conversationId => {
				if (conversationId === "first") {
					firstHandlerStarted.resolve();
					await firstHandlerGate.promise;
				} else if (conversationId === "second") {
					secondHandlerDelivered.resolve();
				}
			},
		});
		harnesses.push(harness);
		const callback = harness.session.agent.telemetry?.onChatUsage;
		if (!callback) throw new Error("Expected SDK to install chat usage telemetry");
		await callback(usage("first"));
		await callback(usage("second"));

		await initializeExtensions(harness.session, {
			reportSendError: () => {},
			reportRuntimeError: () => {},
		});
		await firstHandlerStarted.promise;
		expect(received).toEqual(["first"]);
		firstHandlerGate.resolve();
		await secondHandlerDelivered.promise;
	});

	it("warns with the dropped count when disposed before extension initialization", async () => {
		const received: string[] = [];
		const warnSpy = vi.spyOn(logger, "warn").mockImplementation(() => {});
		const harness = await createHarness({
			received,
			onUsage: () => {},
		});
		harnesses.push(harness);
		const callback = harness.session.agent.telemetry?.onChatUsage;
		if (!callback) throw new Error("Expected SDK to install chat usage telemetry");
		await callback(usage("dropped"));

		await harness.session.dispose();

		const warning = warnSpy.mock.calls.find(
			([message]) => message === "chat_usage pre-initialize buffer cleared during shutdown",
		);
		expect(warning?.[1]).toEqual({ droppedCount: 1 });
		expect(received).toEqual([]);
	});

	it("preserves a previous callback failure while still delivering extension usage", async () => {
		const received: string[] = [];
		const { promise: errorDelivered, resolve: resolveError } = Promise.withResolvers<void>();
		const previous = vi.fn(() => {
			throw new Error("previous telemetry failure");
		});
		const harness = await createHarness({
			received,
			onUsage: previous,
			onReceived: conversationId => {
				if (conversationId === "error") resolveError();
			},
		});
		harnesses.push(harness);
		const callback = harness.session.agent.telemetry?.onChatUsage;
		if (!callback) throw new Error("Expected SDK to install chat usage telemetry");
		await expect(callback(usage("error"))).rejects.toThrow("previous telemetry failure");
		await initializeExtensions(harness.session, {
			reportSendError: () => {},
			reportRuntimeError: () => {},
		});
		await errorDelivered;
		expect(previous).toHaveBeenCalledTimes(1);
		expect(received).toEqual(["error"]);
	});

	it("keeps telemetry disabled when no extension subscribes to chat usage", async () => {
		const received: string[] = [];
		const tempDir = TempDir.createSync("@pi-chat-usage-no-handler-");
		const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
		authStorage.setRuntimeApiKey("openai", "test-key");
		const modelRegistry = new ModelRegistry(authStorage, tempDir.join("models.yml"));
		const cwd = tempDir.path();
		const result = await createAgentSession({
			cwd,
			agentDir: tempDir.path(),
			authStorage,
			modelRegistry,
			sessionManager: SessionManager.inMemory(cwd),
			settings: Settings.isolated({ "async.enabled": false }),
			model: getBundledModel("openai", "gpt-4o-mini"),
			disableExtensionDiscovery: true,
			skills: [],
			contextFiles: [],
			workspaceTree: workspaceTree(cwd),
			promptTemplates: [],
			slashCommands: [],
			enableMCP: false,
			enableLsp: false,
		});
		const harness = { tempDir, authStorage, session: result.session };
		harnesses.push(harness);
		await result.session.agent.telemetry?.onChatUsage?.(usage("ignored"));
		expect(received).toHaveLength(0);
		expect(result.session.agent.telemetry).toBeUndefined();
	});
});
