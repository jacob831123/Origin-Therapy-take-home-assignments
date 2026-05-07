# Referral Inbox Triage Agent

An LLM-powered triage agent for Cedar Kids Therapy that processes a weekend inbox of pediatric therapy referrals, voicemails, portal messages, and emails into a sorted, human-reviewable action plan.

## How to run

```bash
# Install dependencies
npm install

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here

# Run the triage agent (processes inbox, writes output.json and trace)
npm run triage

# Validate the output
npm run validate
```

Custom paths are supported:

```bash
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

End-to-end runtime is approximately 1.5-2 minutes for 8 items (limited by sequential LLM API calls).

## Stack and runtime

- **Language:** TypeScript (strict mode), Node LTS, npm
- **LLM:** Anthropic Claude Sonnet 4 (`claude-sonnet-4-20250514`) via `@anthropic-ai/sdk`
- **Build tooling:** Cursor (AI coding agent used during development)
- **Runtime deps:** `@anthropic-ai/sdk` (LLM client), `ajv`/`ajv-formats` (validation), `ulid` (ID generation), `tsx` (TS execution)
- **No modifications** to `src/tools.ts`, `src/index.ts`, `src/types.ts`, or `src/validate.ts`

**Assumptions:**
- The `ANTHROPIC_API_KEY` environment variable is set before running
- Items are processed sequentially (no parallelism) for trace ordering predictability
- The scenario date is Monday, April 28, 2026 at 8:00 AM (derived from inbox timestamps)

## Architecture

The agent uses a **two-pass LLM architecture** per inbox item:

```
inbox.json (8 items)
  │
  ▼
┌─────────────────────────────────────────────┐
│  For each item (inside withItemContext):     │
│                                             │
│  Pass 1: Classify + Extract + Plan          │
│  ├─ LLM analyzes the inbox item             │
│  ├─ Returns: classification, urgency,       │
│  │   extracted_intake, missing_info,        │
│  │   and a tool_plan (query tools only)     │
│  │                                          │
│  Execute tool_plan                          │
│  ├─ search_patient, verify_insurance,       │
│  │   lookup_policy, find_slots              │
│  │                                          │
│  Pass 2: Synthesize + Act                   │
│  ├─ LLM sees classification + tool results  │
│  ├─ Returns: draft_reply, tasks,            │
│  │   escalation, rationale, hold_slot plan  │
│  │                                          │
│  Execute actions                            │
│  ├─ escalate (if P0/P1)                     │
│  ├─ create_task (collect task_ids)          │
│  ├─ draft_message (if reply needed)         │
│  └─ hold_slot (if in-network + slots found) │
│                                             │
│  Assemble ItemOutput                        │
│  └─ tools_called = getToolCallsForItem()    │
└─────────────────────────────────────────────┘
  │
  ▼
buildBatchOutput() → output.json
```

**Why two passes?**

Pass 1 decides *what to look up*. Pass 2 decides *what to do* based on real tool results. This matters for decision-dependent workflows: e.g., if `verify_insurance` returns out-of-network, Pass 2 knows not to hold a slot and instead creates a billing task for a benefits conversation.

**System prompt** embeds all practice policies (service lines, insurance, safeguarding, clinical advice, scheduling, cancellation, language access), urgency calibration rules, tool schemas with parameter constraints, and explicit behavioral guardrails (no clinical advice, no scheduling, no sending).

**Error handling:** If any item fails (LLM parse error, API timeout), `buildFallbackOutput()` produces a minimal valid output with `classification: "other"`, `urgency: "P2"`, and a rationale explaining the failure. The batch never crashes.

## Triage decisions by item

| Item | Classification | Urgency | Key tools | Rationale |
|------|---------------|---------|-----------|-----------|
| item_1 (Emma Lee) | new_referral | P2 | verify_insurance → in-network, find_slots, hold_slot | Standard SLP referral, complete info, BCBS verified |
| item_2 (Leo) | safeguarding | P0 | lookup_policy(safeguarding), escalate(P0) | "dad getting rough" triggers mandated-reporter protocol |
| item_3 (Owen Brooks) | new_referral | P2 | verify_insurance → out-of-network | Kaiser OON; no slot hold until benefits conversation |
| item_4 (Mateo Ramirez) | new_referral | P2 | search_patient → match, verify_insurance → in-network, find_slots, hold_slot | Existing patient, Aetna verified, PT slot held |
| item_5 (Ava) | clinical_question | P2 | lookup_policy(clinical_advice) | Cannot give clinical advice; routed to clinical lead |
| item_6 (Sam Taylor) | missing_paperwork | P2 | search_patient | Missing DOB, parent, insurance; task to contact referring office |
| item_7 (Isabella Lopez) | new_referral | P2 | verify_insurance → in-network, find_slots(SLP, es), hold_slot | Medicaid verified, Spanish-speaking, draft reply in Spanish |
| item_8 (Noah Patel) | scheduling | P1 | search_patient → match, find_slots(OT) | Same-day cancellation = P1 operational issue |

## Failure modes and production eval

**Known failure modes:**
- **LLM hallucination on tool arguments:** The LLM might generate invalid slot IDs or incorrect date formats. Mitigated by explicit parameter constraints in the system prompt and by passing full tool result data (including real slot IDs) to Pass 2.
- **LLM JSON parse failures:** If the LLM wraps output in markdown fences or adds commentary, the `parseJSON()` function strips fences before parsing. If it still fails, the fallback output kicks in.
- **Urgency miscalibration:** Over-escalation (calling everything P0) or under-escalation (missing safeguarding). Mitigated by embedding the urgency rubric directly in the system prompt with the "default to P2" heuristic.
- **Latency:** Sequential processing of 8 items with 2 LLM calls each takes ~1.5-2 minutes. Acceptable for a Monday-morning batch, but would need parallelization at scale.
- **Non-determinism:** LLM outputs vary between runs. Classifications and tool plans are generally stable, but draft reply wording and task notes will differ.

**Production eval strategy:**
- **Golden-set regression tests:** Maintain a curated set of inbox items with expected classifications, urgency levels, and required tool calls. Run after every prompt or model change.
- **Safeguarding recall metric:** Track false negatives on P0 items specifically. Missing a safeguarding case is the highest-cost failure.
- **Tool call audit:** Alert on items with zero tool calls, or items where tool calls don't match the classification (e.g., a new_referral with no verify_insurance).
- **Human review feedback loop:** Track how often staff override the agent's classification or urgency. High override rates on specific categories signal prompt tuning needed.
- **Draft reply quality:** Sample-review draft replies for tone, accuracy, and absence of clinical advice. Flag replies that mention scheduling as confirmed rather than pending.

## What I chose not to build, and why

- **Parallel item processing:** Items are processed sequentially. Parallelizing would cut runtime from ~2 minutes to ~20 seconds, but adds complexity around trace ordering and error isolation. Not worth it for an 8-item batch under time pressure.
- **Prompt caching / Anthropic beta features:** The system prompt is identical across all 16 API calls and would benefit from prompt caching. Skipped to keep the implementation straightforward.
- **Structured output / tool_use mode:** Claude supports native tool_use for structured output. I used plain JSON-in-text with a `parseJSON` helper instead, which is simpler and avoids provider-specific API surface. The tradeoff is a small risk of parse failures.
- **Confidence scoring:** The agent doesn't output a confidence score per item. In production, low-confidence items could be flagged for priority human review.
- **Input validation:** No validation that inbox items conform to the expected schema before processing. The agent trusts the input format.

## What I would do with another 4 hours

1. **Native tool_use mode:** Switch from JSON-in-text to Claude's native tool_use API. This eliminates JSON parse failures and gives the LLM a structured way to call tools in a loop (agentic tool use), rather than planning all calls upfront.
2. **Parallel processing with concurrency limit:** Process items in parallel (e.g., 4 at a time) with proper error isolation per item. Would bring runtime under 30 seconds.
3. **Prompt caching:** Enable Anthropic's prompt caching for the system prompt to reduce latency and cost by ~80% on input tokens.
4. **Integration tests with variant inputs:** Write a test suite that runs the agent against synthetic variants (different payers, different safeguarding signals, edge cases like duplicate referrals) and asserts on classification, urgency, and required tool calls.
5. **Confidence-based routing:** Add a confidence field to the LLM output. Items below a threshold get flagged for priority human review with a note explaining what the agent was uncertain about.
6. **Observability:** Add structured logging with item_id, LLM latency, token usage, and tool call timing. In production this would feed into dashboards for monitoring agent performance and cost.
