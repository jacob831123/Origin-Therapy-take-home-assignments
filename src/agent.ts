import Anthropic from "@anthropic-ai/sdk";
import type {
  InboxItem,
  ItemOutput,
  ExtractedIntake,
  Discipline,
  Classification,
  Urgency,
  PolicyTopic,
  Assignee,
  ToolResult,
} from "./types.js";
import {
  withItemContext,
  getToolCallsForItem,
  search_patient,
  verify_insurance,
  lookup_policy,
  find_slots,
  hold_slot,
  create_task,
  draft_message,
  escalate,
} from "./tools.js";

const client = new Anthropic();
const MODEL = "claude-sonnet-4-20250514";

function buildSystemPrompt(today: string): string {
  return `You are a triage agent for Cedar Kids Therapy, a multi-disciplinary pediatric therapy practice supporting speech-language pathology (SLP), occupational therapy (OT), and physical therapy (PT). Today is ${today}. You are processing the inbox.

## Practice Policies

### Service Lines
Cedar Kids Therapy serves children ages 0-18 for SLP, OT, and PT. Intake should confirm the requested discipline before scheduling an evaluation.

### Insurance
In-network payers: Aetna, Blue Cross Blue Shield, UnitedHealthcare, and Medicaid.
Out-of-network payers include Kaiser, Cigna Select, and Beacon. Out-of-network referrals require a benefits conversation before a slot is held or recommended as ready to schedule.
Verified status from billing systems supersedes payer information on referral documents. When they conflict, trust the system of record and surface the discrepancy.

### Safeguarding
Any message suggesting harm, abuse, neglect, or unsafe caregiving is P0. Escalate to the clinical lead immediately and create a same-hour review task. Do not provide investigative advice over message. Draft only a neutral acknowledgement for staff review.

### Clinical Advice
Front desk staff and automated systems must not provide clinical advice over message. Clinical questions should be routed to evaluation, screening, or clinician review.

### Scheduling
Same-day cancellations or reschedules are P1 operational issues. Agents may find or hold slots for human review but must not schedule appointments.

### Cancellation
Families should notify the office as soon as possible for same-day illness or cancellation. Makeup visits depend on provider capacity and require staff review.

### Language Access
Families may request communication in Spanish. When possible, match Spanish-speaking families with Spanish-capable staff or providers and draft the response in the family's preferred language.

## Urgency Calibration
- P0: safeguarding, imminent harm, mandated-reporter escalation. Same-hour human review.
- P1: same-day operational issue requiring prompt staff action.
- P2: normal intake, scheduling, billing, or clinical-review workflow.
- P3: low-priority admin, FYI, spam.
Default to P2 unless there is a clear safety or same-day operational reason. Over-escalation is itself a production failure mode.

## Available Tools
You can plan calls to these tools:
1. search_patient({ name?, dob? }) - Search for existing patients by name and/or DOB
2. verify_insurance({ payer?, member_id? }) - Check insurance coverage status
3. lookup_policy({ topic }) - Look up practice policy. Topics: service_lines, insurance, safeguarding, clinical_advice, scheduling, cancellation, language_access
4. find_slots({ discipline?, preferences?, language? }) - Find available appointment slots. IMPORTANT: language must be an ISO code: "en" or "es" (not "English" or "Spanish").
5. hold_slot({ slot_id, patient_ref }) - Place a temporary hold on a slot for human review. IMPORTANT: slot_id must be the exact slot_id string returned by find_slots (e.g. "slot_slp_maya_2026_04_29_0900"), not a datetime or made-up ID.
6. create_task({ assignee, title, due, notes }) - Create a task. Assignees: front_desk, intake, billing, clinical_lead. Due dates must be YYYY-MM-DD format and realistic relative to today (${today}).
7. draft_message({ recipient, channel, body, language? }) - Draft a message (NOT send). Channels: portal, email, phone. Language: en or es
8. escalate({ item_id, reason, severity }) - Escalate an item. Severity: P0 or P1

## Constraints
- Every item requires human review (requires_human_review = true always).
- Do NOT provide clinical advice in draft replies.
- Do NOT schedule appointments. Only find_slots and hold_slot (pending review).
- Do NOT auto-send messages. Only draft_message.
- Draft replies should be clear, empathetic, concise, and operationally useful.
- Tool calls must be purposeful and relevant to the item. Do not make performative calls.`;
}

function deriveToday(inbox: InboxItem[]): { display: string; iso: string } {
  const latest = inbox
    .map((item) => new Date(item.received_at))
    .reduce((a, b) => (a > b ? a : b));
  const display = latest.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "America/Los_Angeles",
  });
  const iso = latest.toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
  return { display, iso };
}

interface ToolPlan {
  name: string;
  args: Record<string, unknown>;
}

interface Pass1Result {
  classification: Classification;
  urgency: Urgency;
  extracted_intake: ExtractedIntake;
  missing_info: string[];
  tool_plan: ToolPlan[];
}

interface Pass2Result {
  recommended_next_action: string;
  draft_reply: string | null;
  draft_reply_recipient: string | null;
  draft_reply_channel: "portal" | "email" | "phone";
  draft_reply_language: "en" | "es";
  decision_rationale: string;
  escalation_reason: string | null;
  escalation_severity: "P0" | "P1" | null;
  tasks: Array<{
    assignee: Assignee;
    title: string;
    due: string;
    notes: string;
  }>;
  additional_tools: ToolPlan[];
}

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const today = deriveToday(inbox);
  const systemPrompt = buildSystemPrompt(today.display);
  console.log(`Triage date: ${today.display} (${today.iso})`);

  const results: ItemOutput[] = [];

  for (const item of inbox) {
    console.log(`Processing ${item.id}: ${item.subject}`);
    try {
      const output = await processItem(item, systemPrompt, today.iso);
      results.push(output);
      console.log(`  -> ${output.classification} / ${output.urgency} / ${output.tools_called.length} tool calls`);
    } catch (err) {
      console.error(`  -> Error processing ${item.id}:`, err);
      results.push(buildFallbackOutput(item, err));
    }
  }

  return results;
}

async function processItem(item: InboxItem, systemPrompt: string, todayISO: string): Promise<ItemOutput> {
  return withItemContext(item.id, async () => {
    const pass1 = await classifyAndPlan(item, systemPrompt);

    // Execute planned tool calls (except create_task, draft_message, escalate — those come from Pass 2)
    const toolResults = await executeToolPlan(pass1.tool_plan, item);

    const pass2 = await synthesize(item, pass1, toolResults, systemPrompt, todayISO);

    // Execute Pass 2 actions: escalation, tasks, draft message
    const taskIds: string[] = [];

    if (pass2.escalation_reason && pass2.escalation_severity) {
      await escalate({
        item_id: item.id,
        reason: pass2.escalation_reason,
        severity: pass2.escalation_severity,
      });
    }

    for (const t of pass2.tasks) {
      const result = await create_task(t);
      taskIds.push(result.data.task_id);
    }

    if (pass2.draft_reply && pass2.draft_reply_recipient) {
      await draft_message({
        recipient: pass2.draft_reply_recipient,
        channel: pass2.draft_reply_channel,
        body: pass2.draft_reply,
        language: pass2.draft_reply_language,
      });
    }

    // Execute any additional tools from Pass 2
    if (pass2.additional_tools?.length) {
      await executeToolPlan(pass2.additional_tools, item);
    }

    return {
      item_id: item.id,
      classification: pass1.classification,
      urgency: pass1.urgency,
      requires_human_review: true,
      extracted_intake: pass1.extracted_intake,
      missing_info: pass1.missing_info,
      tools_called: getToolCallsForItem(item.id),
      recommended_next_action: pass2.recommended_next_action,
      draft_reply: pass2.draft_reply,
      task_ids: taskIds,
      escalation:
        pass2.escalation_reason && pass2.escalation_severity
          ? {
              reason: pass2.escalation_reason,
              severity: pass2.escalation_severity,
            }
          : null,
      decision_rationale: pass2.decision_rationale,
    };
  });
}

async function classifyAndPlan(item: InboxItem, systemPrompt: string): Promise<Pass1Result> {
  const userPrompt = `Analyze this inbox item and return a JSON object.

## Inbox Item
- ID: ${item.id}
- Channel: ${item.channel}
- Received: ${item.received_at}
- Sender: ${item.sender}
- Subject: ${item.subject}
- Body: ${item.body}
- Attachments: ${item.attachments.length > 0 ? item.attachments.join(", ") : "none"}

## Instructions
Return a JSON object with these fields:
{
  "classification": one of "new_referral", "existing_patient_request", "scheduling", "clinical_question", "billing_question", "missing_paperwork" (use for referrals missing critical info like DOB, parent contact, or insurance), "provider_followup", "complaint", "safeguarding", "spam", "other",
  "urgency": one of "P0", "P1", "P2", "P3",
  "extracted_intake": {
    "child_name": string or null,
    "dob_or_age": string or null (use DOB format YYYY-MM-DD if available, otherwise age like "6 years"),
    "parent_contact": string or null (include name, phone, email as available),
    "discipline": array of "SLP"/"OT"/"PT" or null,
    "diagnosis_or_concern": string or null,
    "payer": string or null,
    "member_id": string or null
  },
  "missing_info": array of strings describing what key info is missing (empty if complete),
  "tool_plan": array of tool calls to execute for information gathering. Each: { "name": string, "args": object }
}

For tool_plan, include ONLY information-gathering tools:
- search_patient: if there's a child name + DOB to look up
- verify_insurance: if there's a payer to verify
- lookup_policy: if a policy topic is relevant to the decision
- find_slots: if the item needs scheduling and insurance is expected to be in-network

Do NOT include create_task, draft_message, escalate, or hold_slot in tool_plan — those will be decided in a later step after seeing tool results.

Return ONLY the JSON object, no markdown fences or extra text.`;

  const text = await callLLM(systemPrompt, userPrompt, 1024, "Pass1");
  return parseJSON<Pass1Result>(text, "Pass1");
}

async function executeToolPlan(
  plan: ToolPlan[],
  item: InboxItem,
): Promise<Map<string, ToolResult<unknown>>> {
  const results = new Map<string, ToolResult<unknown>>();

  for (const call of plan) {
    try {
      const result = await executeOneTool(call, item);
      if (result) {
        results.set(`${call.name}:${JSON.stringify(call.args)}`, result);
      }
    } catch (err) {
      console.error(`  Tool ${call.name} failed:`, err);
    }
  }

  return results;
}

async function executeOneTool(
  call: ToolPlan,
  item: InboxItem,
): Promise<ToolResult<unknown> | null> {
  switch (call.name) {
    case "search_patient":
      return search_patient(call.args as { name?: string; dob?: string });
    case "verify_insurance":
      return verify_insurance(
        call.args as { payer?: string; member_id?: string },
      );
    case "lookup_policy":
      return lookup_policy(call.args as { topic: PolicyTopic });
    case "find_slots":
      return find_slots(
        call.args as {
          discipline?: Discipline;
          preferences?: string;
          language?: string;
        },
      );
    case "hold_slot":
      return hold_slot(
        call.args as { slot_id: string; patient_ref: string },
      );
    case "create_task":
      return create_task(
        call.args as {
          assignee: Assignee;
          title: string;
          due: string;
          notes: string;
        },
      );
    case "draft_message":
      return draft_message(
        call.args as {
          recipient: string;
          channel: "portal" | "email" | "phone";
          body: string;
          language?: "en" | "es";
        },
      );
    case "escalate":
      return escalate(
        call.args as { item_id: string; reason: string; severity: "P0" | "P1" },
      );
    default:
      console.warn(`  Unknown tool: ${call.name}`);
      return null;
  }
}

async function synthesize(
  item: InboxItem,
  pass1: Pass1Result,
  toolResults: Map<string, ToolResult<unknown>>,
  systemPrompt: string,
  todayISO: string,
): Promise<Pass2Result> {
  const toolSummaries = Array.from(toolResults.entries())
    .map(([_key, result]) => `- ${result.name}(${JSON.stringify(result.args)}): ${result.result_summary}\n  Full result: ${JSON.stringify(result.data)}`)
    .join("\n");

  const userPrompt = `Based on the triage analysis and tool results, produce the final action plan.

## Inbox Item
- ID: ${item.id}
- Channel: ${item.channel}
- Sender: ${item.sender}
- Subject: ${item.subject}
- Body: ${item.body}

## Classification
- Classification: ${pass1.classification}
- Urgency: ${pass1.urgency}
- Missing info: ${pass1.missing_info.length > 0 ? pass1.missing_info.join("; ") : "none"}

## Tool Results
${toolSummaries || "No tools were called."}

## Instructions
Return a JSON object with these fields:
{
  "recommended_next_action": concise string describing what staff should do next,
  "draft_reply": string with the draft reply to send to the parent/sender, or null if no reply is appropriate. Must be empathetic, concise, and operationally useful. Must NOT contain clinical advice. If the family speaks Spanish, draft in Spanish.
  "draft_reply_recipient": the email/phone/portal ID to send to, or null,
  "draft_reply_channel": "portal" or "email" or "phone",
  "draft_reply_language": "en" or "es",
  "decision_rationale": 1-3 sentence explanation of why this classification/urgency/action was chosen,
  "escalation_reason": string reason if this needs P0/P1 escalation, or null,
  "escalation_severity": "P0" or "P1" if escalation needed, or null,
  "tasks": array of tasks to create. Each: { "assignee": "front_desk"|"intake"|"billing"|"clinical_lead", "title": string, "due": "YYYY-MM-DD", "notes": string },
  "additional_tools": array of additional tool calls if needed (e.g. hold_slot after finding slots). Each: { "name": string, "args": object }. Only include find_slots or hold_slot here if appropriate and not already called.
}

Guidelines for tasks (today is ${todayISO}):
- For P0 safeguarding: create a task for clinical_lead with same-day due date (${todayISO})
- For new referrals with in-network insurance: create a task for intake, due within 1-2 business days
- For out-of-network: create a task for billing, due within 1-2 business days
- For incomplete referrals: create a task for front_desk, due within 1-2 business days
- For same-day reschedule: create a task for front_desk with same-day due date (${todayISO})
- For clinical questions: create a task for clinical_lead, due within 2-3 business days

Guidelines for draft replies:
- Address the parent/sender by name when possible
- Be warm but professional
- Never promise appointments or provide clinical advice
- For safeguarding: draft a neutral acknowledgement only, for staff review
- For Spanish-speaking families: draft in Spanish

Guidelines for hold_slot in additional_tools:
- Only hold a slot if insurance is verified in-network and slots were found
- Use the earliest appropriate slot_id from the find_slots results
- Use the child's name as patient_ref

Return ONLY the JSON object, no markdown fences or extra text.`;

  const text = await callLLM(systemPrompt, userPrompt, 1500, "Pass2");
  return parseJSON<Pass2Result>(text, "Pass2");
}

function buildFallbackOutput(item: InboxItem, err: unknown): ItemOutput {
  return {
    item_id: item.id,
    classification: "other",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: {
      child_name: null,
      dob_or_age: null,
      parent_contact: null,
      discipline: null,
      diagnosis_or_concern: null,
      payer: null,
      member_id: null,
    },
    missing_info: ["Processing failed — requires manual review"],
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action:
      "Manual review required — automated processing encountered an error.",
    draft_reply: null,
    task_ids: [],
    escalation: null,
    decision_rationale: `Automated processing failed: ${err instanceof Error ? err.message : String(err)}. Item requires full manual review.`,
  };
}

async function callLLM(
  system: string,
  userPrompt: string,
  maxTokens: number,
  label: string,
): Promise<string> {
  const MAX_ATTEMPTS = 3;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const response = await client.messages.create({
        model: MODEL,
        max_tokens: maxTokens,
        system,
        messages: [{ role: "user", content: userPrompt }],
      });
      return extractText(response);
    } catch (err) {
      const status = err instanceof Error && "status" in err
        ? (err as { status: number }).status
        : 0;
      const isRetryable = status === 429 || status >= 500;
      if (!isRetryable || attempt === MAX_ATTEMPTS) throw err;
      console.warn(`  ${label} attempt ${attempt} failed (${status}), retrying...`);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
  throw new Error("unreachable");
}

function extractText(response: Anthropic.Message): string {
  for (const block of response.content) {
    if (block.type === "text") {
      return block.text;
    }
  }
  throw new Error("No text content in LLM response");
}

function parseJSON<T>(text: string, label: string): T {
  let cleaned = text.trim();
  // Strip markdown code fences if present
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }
  try {
    return JSON.parse(cleaned) as T;
  } catch (err) {
    console.error(`Failed to parse ${label} JSON:`, cleaned);
    throw new Error(
      `Failed to parse ${label} response as JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}
