# capability-runtime

[![tests](https://github.com/HarshitGadge/capability-runtime/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshitGadge/capability-runtime/actions/workflows/tests.yml)

An LLM discovers a flow on a live application once. The run is frozen into a typed, versioned
**capability artifact**. That artifact then replays **deterministically, with no model in the
loop** — returning declared outputs, distinguishing legitimate business answers from real
failures, recovering from anticipated interruptions, and handing control to a human when it must
not proceed alone.

> The model discovers. The artifact becomes a reusable capability. Deterministic replay is how an
> agent invokes it in production.

Design reasoning, trade-offs and cut lines: **[REPORT.md](./REPORT.md)**.

---

## How it fits together

```mermaid
flowchart LR
    subgraph discovery["Discovery — once, with a model"]
        LLM["LLM agent<br/>observe → decide → act"]
        REC["recording"]
        LLM --> REC
    end
    REC --> COMP["compiler<br/>ranked locators,<br/>checkpoints, redaction"]
    COMP --> ART["Capability artifact<br/>(typed · versioned · reviewable)"]

    subgraph production["Production — many times, no model"]
        AGENT["AI agent<br/>(agent-facing product)"]
        REPLAY["replay engine<br/>observe → verify → act"]
        AGENT -- "invoke(inputs)" --> REPLAY
        REPLAY -- "Success · BusinessOutcome<br/>Failure · Escalated" --> AGENT
    end
    ART --> REPLAY

    REPLAY -. "stuck / irreversible" .-> HUMAN["human operator<br/>(same live session)"]
    HUMAN -. "authorize · resume" .-> REPLAY

    subgraph surface["Surface (one interface, swappable)"]
        S1["browser · DOM-scan"]
        S2["browser · CDP a11y tree"]
        S3["desktop · UIA / AX (design)"]
    end
    REPLAY --> surface
    GATE["policy gate: allowlist · risk ceiling · redaction · control lease"]
    REPLAY -.-> GATE
```

Everything above the Surface interface — the agent loop, the artifact, the replay engine, the
policy gate — is written against a platform-independent `UiElement`, never a DOM or a selector.
Two browser perception providers ship today; a desktop one is a design, not a build.

---

## Setup

Requires Node 20+.

```bash
npm install
npx playwright install chromium
```

A model API key is needed **only** for the discovery run. Replay, the tests and every other demo
below run without one.

```bash
cp .env.example .env      # then set ANTHROPIC_API_KEY
```

The stand-in portal needs no credentials — it is a local mock with synthetic data. The service
sign-on the replay engine performs uses `PORTAL_USER` / `PORTAL_PASS`, which default to the mock's
throwaway values so nothing is required to run the demo.

---

## The target application

`app/` serves a deliberately hostile stand-in for a credit-union member-services portal: content
two frames deep, nested table layout, inline handlers, **no ids and no test hooks**, and inputs
whose only label is the text in the adjacent table cell. It is served under two tenant skins —
`tenant-a` (Northgate Credit Union) and `tenant-b` (Vale Mutual Savings) — from one codebase, which
is the shape of the real problem: many institutions running one vendor product.

All data is synthetic. No real members, balances or credentials.

```bash
npm run app        # http://localhost:5173
```

| Member  | Result when looked up                          |
| ------- | ---------------------------------------------- |
| `12345` | the happy path                                 |
| `10001` | another valid member                           |
| `55501` | flagged restricted → `ACCESS_RESTRICTED`       |
| `99999` | no such record → `MEMBER_NOT_FOUND`            |

---

## Demo path

**Leave `npm run app` running in another terminal for everything below.**

### 1. Discover a capability with a real LLM run

```bash
npm run discover -- --goal "Look up member 12345 and read their current savings balance"
```

The agent explores the portal, then records the flow cleanly, and the run is compiled into
`artifacts/member.read_savings_balance.json`. Options: `--entry <url>` for a different entry
point, `--headless`, `--provider cdp` to perceive through the browser's real accessibility tree
instead of the in-page projection, `--max-steps` / `--timeout-ms` (also `DISCOVERY_MAX_STEPS`,
`DISCOVERY_TIMEOUT_MS` in `.env`), and `--allow-risk irreversible` for a supervised recording of
a flow that submits something irreversible.

The run stops on one of five reasons — `goal_reached`, `max_steps`, `timeout`, `dead_end` (the
same action repeated, or the screen unchanged across several actions), or `model_declined` — and
exits non-zero for anything but the first.

Everything from the run lands in `evidence/discovery-<id>/` — the structured log, the full model
transcript, per-turn screenshots, and `recording.json`. Three real runs are committed — the balance lookup (14 turns, ~$0.32), the sub-account opening
(27 turns, an irreversible step), and a funds transfer (32 turns, `INSUFFICIENT_FUNDS` discovered
by probing). See `evidence/README.md`, including the run the model declined and how it was handled.

### 2. Replay it deterministically

```bash
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345
```

```
RESULT: SUCCESS
  outputs   {"savings_balance":18204.37,"savings_account":"4718355901","member_name":"J. Whitfield","status":"active"}

  ✓ enter_the_member_id_into_the_lookup_field         proximity_label(textbox, "Member ID")
  ✓ submit_the_member_search                          role_name(button, "Search")
  ✓ open_the_matching_member_record_from_the_search   role_name(link, "View")
  ✓ capture_the_savings_balance_...                   proximity_label(cell, "Savings Balance")
  ✓ capture_the_savings_account_number_...            proximity_label(cell, "Savings Account")
  ✓ capture_the_member_name_...                       proximity_label(cell, "Member Name")
  ✓ capture_the_member_status_...                     proximity_label(cell, "Status")
```

The model chose to extract four outputs where the goal asked for one; the artifact declares
each with a type and a sensitivity, and the three regulated ones are withheld from the on-disk
log while being returned to the caller.

You will also see this on the same run:

```
  drift     surface fingerprint changed (recorded 9677b7…, observed 5610be…) — re-review this capability for this tenant
              open_the_matching_member_s_detail_record: /member+/nav:5 → /member+/nav:6
              capture_the_savings_balance_from_the_mem: /member+/nav:5 → /member+/nav:6
              ...
```

That is not a bug — it is the drift detector catching a real change. This capability was recorded
*before* the "Transfer Funds" link was added to the member screen; every step on that screen now
sees six controls where the recording saw five. The capability still succeeds, because targeting is
semantic rather than positional, and the run tells a reviewer exactly which step to look at and
what changed. Drift never fails a run — the checkpoints already decided that — it flags the
capability for review. It reports identically through both perception providers, because it
measures the app, not the lens.

No API key is used. The rightmost column is which rung of the locator cascade actually matched —
targeting degradation is reported, never silent. Every result also carries a **surface
fingerprint** comparison; a `drift` line appears when the screens the flow passed through have
changed shape since the recording.

```bash
# Same artifact, perceived through the browser's accessibility tree instead of the in-page scanner
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 --provider cdp
#   RESULT: SUCCESS — same rungs, and the same drift report. Nothing above `Surface` knows which one ran.
```

Exit codes let a caller branch without parsing output: **0** success · **3** business outcome ·
**4** escalated · **1** failure.

### 3. Exercise the error paths

```bash
# A legitimate business answer, not a crash
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=99999
#   RESULT: BUSINESS_OUTCOME   code MEMBER_NOT_FOUND

# A different business answer, told apart from the one above
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=55501
#   RESULT: BUSINESS_OUTCOME   code ACCESS_RESTRICTED

# Rejected by the contract before a page is opened
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=abc
#   RESULT: FAILURE   kind invalid_input

# A blocking interstitial appears mid-flow and is recovered from
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 --inject interstitial
#   RESULT: SUCCESS   ↻ [recovered: MAINTENANCE_INTERSTITIAL]

# The session is dropped mid-flow: re-authenticate and restart the flow
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 --inject session_timeout
#   RESULT: SUCCESS   ↻ [recovered: SIGNED_OUT]   (earlier steps visibly re-run)

# A slow response, absorbed with no fixed sleep anywhere in the system
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 --inject slow
```

`--inject` arms a fault in the stand-in portal. It is a test-harness control plane
(`POST /__control`), not part of the modelled application.

### 3b. A third capability: funds transfer, with an unexpected dialog

```bash
T=artifacts/member.post_savings_transfer.json
npm run replay -- --artifact $T --input member_id=10002 --input to_account=999888777 --input amount=99999.00
#   RESULT: BUSINESS_OUTCOME   code INSUFFICIENT_FUNDS
npm run replay -- --artifact $T --input member_id=12345 --input to_account=999888777 --input amount=5000.00
#   ↻ [recovered: HIGH_VALUE_CONFIRM]   ← a confirmation dialog the recording never saw
#   RESULT: ESCALATED at confirm_and_post_the_transfer   ← the irreversible step routes to a human
```

The transfer was recorded against a $50 amount, so its recording contains no dialog. High-value
transfers raise one at runtime; the app-level `HIGH_VALUE_CONFIRM` recovery handles it and the flow
continues to the irreversible confirm, which escalates. Every capability inherits app-level
recoveries without re-recording.

### 4. Run the same artifact against a second institution

```bash
# Unspecialized against a new skin: fails, and names exactly what drifted
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 \
  --tenant tenant-b --no-overlay
#   RESULT: FAILURE   kind precondition_unmet
#   expected  text present: "Member Lookup"
#   remediation  If the tenant renamed this control, add a targetOverride to that tenant's overlay

# With a three-line tenant overlay: the same artifact, reused
npm run replay -- --artifact artifacts/member.read_savings_balance.json --input member_id=12345 \
  --tenant tenant-b
#   RESULT: SUCCESS   outputs {"savings_balance":18204.37, ...}
#   ✓ ... proximity_label(textbox, "Member Number")
#   ✓ ... role_name(button, "Find Member")
```

The overlay is `artifacts/overlays/tenant-b.member.read_savings_balance.json`. It renames two
controls and one screen title, and binds the tenant's origin — the run header shows
`origin ... (bound from overlay tenant-b)`. The step sequence, checkpoints, business outcomes,
recoveries and extraction are reused unchanged.

### 5. Human handoff on an irreversible step

![Operator console mid-escalation](docs/operator-console.png)


```bash
npm run demo:handoff
```

Scripted end-to-end: starts a shared browser session, replays a capability whose final step opens
an account, watches it refuse to proceed unattended, then posts the same form the operator
console's buttons post — claim control, authorize the step once, hand control back — and shows the
run resume and finish.

To drive it by hand instead, in four terminals:

```bash
npm run app                 # 1. the target application
npm run session             # 2. the shared browser — outlives every run
npm run operator            # 3. http://localhost:7788
npm run replay -- --artifact artifacts/member.open_sub_account.json \
  --input member_id=12345 --input account_type="Holiday Savings" --input opening_deposit=250.00
```

The run reaches the account-opening step, classifies it `irreversible`, raises an intervention and
**cedes the session**. Open the console: the request carries the step, the reason, a screenshot, a
redacted screen summary and the CDP endpoint of the live session. Take control — you are driving
the same window the automation was in, and the policy gate now blocks automation entirely. Then
either **Authorize this step** (automation performs it once) or **I did it myself** (automation
skips it). Control returns, the engine re-verifies the step's precondition, and the run completes.

### 6. Let an AI agent decide and call it

```bash
npm run agent -- --task "A member called in — what is the savings balance for member 12345?"
npm run agent -- --task "Can you confirm whether member 99999 exists in our system?"
```

This is the production shape: an LLM is given the capabilities as tools and a natural-language
task, and it decides which to call. The model never sees the portal — deterministic replay runs
underneath each tool call, and the model reasons only about the task and the typed result.

```
TASK: A member called in — what is the savings balance for member 12345?
  → agent calls member_read_savings_balance({"member_id":"12345"})
  ← returned: success {"savings_balance":18204.37, ...}
AGENT ANSWER: The current savings balance for member 12345 (J. Whitfield) is $18,204.37.

TASK: Can you confirm whether member 99999 exists?
  → agent calls member_read_savings_balance({"member_id":"99999"})
  ← returned: business_outcome MEMBER_NOT_FOUND
AGENT ANSWER: No — member 99999 does not exist. The lookup returned MEMBER_NOT_FOUND.
```

The agent relays the business outcome as a plain answer, not an error — because the tool contract
told it that outcome is a legitimate result. Needs the model key (the agent decides); the
capabilities it invokes do not.

### 7. The tool contract, directly

```bash
npm run schema                                              # JSON Schema for artifact, overlay, result
npm run schema -- --tool artifacts/member.read_savings_balance.json
```

The second renders a capability as a tool definition another agent can be handed: declared inputs
become the parameter schema, declared outputs and business-outcome codes go in the description.
The programmatic half is `src/index.ts`:

```ts
import { toolDefinition, invoke } from './src/index.js';

const tool = toolDefinition(artifact);          // what the agent is shown
const result = await invoke({                   // what runs when it calls the tool
  artifact, inputs: { member_id: '12345' }, tenantId: 'tenant-b', baseUrl, overlay,
});
switch (result.status) {                        // what it gets back
  case 'success':          /* result.outputs.savings_balance */ break;
  case 'business_outcome': /* result.code === 'MEMBER_NOT_FOUND' */ break;
  case 'escalated':        /* result.interventionId, result.resumeToken */ break;
  case 'failure':          /* result.kind, expected, observed, screenshotPath */ break;
}
```

---

## Tests

```bash
npm test           # 84 tests; boots the app itself, no API key, no network
npm run test:unit  # the fast ones only (~0.3s)
npm run typecheck
```

Unit tests cover the locator cascade, redaction and the artifact leak guard, the policy gate, risk
classification, overlay merging, the checkpoint language, and the engine's handling of ambiguity,
recovery exhaustion, step timeouts and fingerprint drift. Browser-backed suites spawn the portal
and assert the behaviour the demo path shows — success and repeatability, both business outcomes,
input rejection, all three recovery paths, both perception providers, drift detection and overlay
reuse, human-action capture across frames and navigations during a handoff, and that regulated
values are absent from disk while still being returned to the caller. `test/discovery.test.ts`
drives the real discovery loop with a scripted fake model against the real portal — explore,
declare an outcome, record, compile, and replay the result to both success and `MEMBER_NOT_FOUND`
— so the core loop is tested end to end without an API key.

---

## Running without the model

```bash
npm run fixtures   # drive the flows from a fixed script → evidence/fixture-*/recording.json
npm run compile -- --recording evidence/fixture-read-balance/recording.json
```

`compile` rebuilds an artifact from a saved recording with no API call and no browser. It works on
a real discovery run's `recording.json` too — the artifact is a *derived* document, and being able
to regenerate it offline is both the proof that it is decoupled from the model transcript and the
reason iterating on locator strategy costs nothing. It earned its keep during this project: the
first real run exposed two compiler bugs, and both fixes were verified by recompiling the saved
recording rather than paying for another run.

`compile` is also where a **reviewer** shapes a capability before it ships, with the decision
recorded in provenance:

```bash
npm run compile -- --recording evidence/discovery-<id>/recording.json   --raise member_name=regulated \                       # sensitivity may be raised, never lowered
  --outcome 'ACCESS_RESTRICTED=Authorization required'  # an outcome the model never encountered
```

The shipped artifacts are compiled from the real discovery runs in `evidence/`, with exactly the
reviewer flags noted in each artifact's `provenance.notes`. The scripted fixtures exist so the
**tests** need no model.

---

## Layout

```
app/                    stand-in portal: two tenants, injectable faults
profiles/portal.json    app profile — allowlist, recoveries, session establishment
src/schema/             the artifact, overlay, result and observation contracts (Zod)
src/surface/            Surface interface, browser implementation, the two perception
                        providers, and the locator cascade
src/policy/             redaction, risk classification, and the policy gate
src/agent/              the LLM discovery loop, its tools, the recording → artifact compiler,
                        and invokeAgent — an LLM calling capabilities as tools
src/replay/             the deterministic engine, checkpoints, preflight, overlay merge
src/escalation/         session lifetime, control lease, intervention inbox, action capture
src/evidence/           structured run logs, screenshots, observation snapshots
src/cli/                app · session · discover · replay · agent · operator · compile · schema
artifacts/              capability artifacts, tenant overlays, generated JSON Schema
evidence/               discovery and replay runs
```
