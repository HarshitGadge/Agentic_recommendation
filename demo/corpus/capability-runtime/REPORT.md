# Design write-up

An LLM discovers a flow on a live application once. The run is frozen into a typed, versioned
**capability artifact**. The artifact then replays with **no model in the loop**, returning
declared outputs and distinguishing three things production actually throws at it: business
answers the caller did not want, interruptions that can be handled, and situations that need a
person.

The target is a self-built, deliberately hostile legacy portal — content two frames deep, nested
table layout, no ids or test hooks, inputs whose only label is the text in the adjacent cell —
served under two tenant skins. Built rather than borrowed because it is the only way to trigger
every condition in §3.3 deterministically, on demand, and under a second institution's branding.

## 1. Architecture

```
Surface (interface)      observe() → Observation · resolve(target) · click/type/select/press/navigate
 ├─ BrowserSurface       finds semantically; acts through real mouse and key events
 │   ├─ DomScanProvider  in-page accessibility projection (default)
 │   └─ CdpAxProvider    the browser's real accessibility tree
 └─ PolicyGate           decorator: allowlist · risk ceiling · redaction · control lease

Discovery  LLM loop → recording.json → compiler → CapabilityArtifact (Zod, semver)
Replay     artifact + inputs → Success | BusinessOutcome | Failure | Escalated
Escalation lease + inbox + operator console, over a browser session that outlives the runner
Evidence   JSONL + screenshots + observation snapshots, redacted on the way out
```

**Perception is accessibility-shaped, not DOM-shaped.** Nothing above `Surface` mentions a
browser, a selector or markup; everything is written against `UiElement { role, name, value,
proximityLabel, bounds, frame }`, the tuple UIA or macOS AX produces. There is no `evaluate()`
escape hatch — if a capability needed one, the abstraction would be a lie.

**Two perception providers, because a seam nobody has crossed is a claim.** `--provider cdp`
replays the same artifact through `Accessibility.getFullAXTree` — same four rungs, same
fingerprint. It did not start that way: a raw AX tree gives a bare legacy input no accessible
name, and the first run through it degraded to `ordinal` at step one and failed at step three
(the trace said exactly that). The fix was to derive "the label in the cell to the left" from the
tree's own row/cell structure — what UIA's Table pattern would give a desktop provider — rather
than from the DOM. The lower rungs of the cascade stay, because the next surface will lack
something else.

**The policy gate is a decorator, not a checklist.** Discovery and replay hold a `PolicyGate`
and neither holds the surface beneath it, so no code path can act while bypassing policy. A
guardrail you have to remember to call is not a guardrail.

**The model decides what to do; the runtime decides how to find it again.** The model picks
controls by reference number; the compiler derives targeting from the element actually touched and
keeps only strategies verified to match it uniquely on the recorded screen. An ambiguous-at-record
locator is discarded, not shipped as a latent flake.

**Exploration and recording are separate phases.** The first pass through an unfamiliar UI is
full of back-tracking; the agent explores, calls `start_recording`, the session resets, and it
performs the flow once cleanly. Only that pass becomes steps. In the three real runs the model did
this unprompted — reached the goal, went back to probe an invalid ID, declared the outcome from
what it saw, then recorded (14 turns, ~$0.30 for the balance lookup).

**The real run found two compiler bugs, and the recording paid for neither fix.** The model
nominated "Member Detail — 12345" as its success text, and the compiler shipped it verbatim — a
capability that would only ever pass for one member. And because the model picks elements from
the *redacted* screen, the compiler's uniqueness check compared a token against real text and
silently discarded every semantic rung on the regulated extractions, leaving coordinates. Both
were fixed and verified by recompiling the saved recording offline, which is the point of the
artifact being a derived document.

**The through-line is demonstrated, not asserted.** `npm run agent` gives a real LLM a
natural-language task and the capabilities as tools; it decides which to call, and deterministic
replay runs underneath — the model never sees the portal. For a missing member it relays
`MEMBER_NOT_FOUND` as a plain answer, because the tool contract told it that outcome is a result,
not an error. `src/index.ts` is the seam: `toolDefinition(artifact)` is what the agent is shown,
`invoke(...)` is what runs.

**Trade-off:** one process, files, no queue. The lease is three fields that would become a row
with a TTL; the inbox a directory that would become a table. None of that plumbing is built.

## 2. Artifact schema

Zod in `src/schema/` yields the validator, the types and the JSON Schema from one definition.
`npm run schema -- --tool <artifact>` renders a capability as a tool definition — inputs become
the parameter schema, outputs and business-outcome codes go in the description — which is the
payoff of a typed contract over a step list.

```
CapabilityArtifact
  capability        id · semver · description       ← what a calling agent reads
  target            appId · appVersion · surface · entryPoint · allowlist
  inputs[]          name · type · pattern · sensitivity · example
  outputs[]         name · type · sensitivity · required
  steps[]           intent · action · target · risk · precondition · postcondition · timeoutMs
  success           Checkpoint
  businessOutcomes  code · detect:Checkpoint          ← expected non-success results
  recoveries        code · detect · actions · maxAttempts · then · escalateOnExhaustion
  provenance        model · run · tenant · evidence · surfaceFingerprint
```

**Targets are ranked strategy lists.** `role_name → proximity_label → placeholder → text →
ordinal → coordinates`, with an optional scope and frame path. The top rungs describe what a
control *means* and survive restyling, DOM reordering and tenant skinning; the bottom rungs
describe where it was and survive almost nothing. Replay stops at the first rung that matches and
**records which one won**, so degradation is a visible signal, never a silent flake.
`proximity_label` is not a convenience: this portal's member-ID input has no accessible name at
all. Coordinates are recorded but never choose an element on their own. `ambiguity` says what to
do with more than one match: fail, take the first, or escalate.

**Scopes make row selection stable.** `row_containing: {input: member_id}` means "the View link
in the row for the member we asked about"; `ordinal[2]` means "the third one", true until the
result set changes.

**Business outcomes are declared.** "No such member" is an answer the caller asked for, not an
exception. The artifact carries the detector; replay recognises it by what is on screen and
returns `{status:'business_outcome', code:'MEMBER_NOT_FOUND'}`. The model declares what it
probed; what it did not probe — a restricted record — replays as a hard failure with the screen
text in `observed`, and a reviewer promotes it with `--outcome`, a three-field edit to a
reviewable document rather than a re-record.

**Values are typed references** — `input | extracted | const | secret` — so a reviewer sees what
the caller supplies and what comes from the environment; credentials never enter the document.
**Deliberately absent:** screenshot hashes, DOM-shape assertions, embedded code. Checkpoints are a
small declarative language a non-programmer can review and the engine evaluates without `eval`.

## 3. Determinism & error handling

Replay imports no model client. **One waiting primitive:** poll until a stated checkpoint holds
or time out naming the clause that never became true. No fixed sleeps, no `networkidle`. A slow
surface and a broken one are then distinguishable, which is the difference between a retry and a
failure. Each step also carries a wall-clock `timeoutMs` around the action itself.

**Per-step proof.** Each step has a precondition and postcondition generated from the recorded
screens. Two subtleties: on a frameset the top URL never changes, so `url_matches` evaluates
against every frame; and "first heading on the page" is the navigation menu, identical everywhere,
so the compiler identifies the working frame as the one whose content varies across the recording.

**The classification order is the whole difficulty.** After every step:

1. **Declared business outcome?** Terminate and report it — checked *first*, before any retry.
   Checking it last is the common bug: the engine burns its retry budget on a screen that is
   clearly saying there is no such member.
2. **Declared recovery condition?** Remediate within that rule's budget, then **re-verify the
   postcondition before redoing anything**. Dismissing an interstitial usually reveals the screen
   the step wanted; repeating the action there is a duplicate submission waiting to happen. A rule
   whose budget is exhausted either escalates (`escalateOnExhaustion`) or fails.
3. **Postcondition holds?** Continue.
4. **Otherwise:** hard failure with step, expected, observed, remediation hint, screenshot and
   observation snapshot.

The result is a discriminated union — `success | business_outcome | failure | escalated` — with
exit codes 0/3/4/1 so a caller branches without parsing text. Every arm reports which locator
rung each step used and a **surface fingerprint** comparison: the sequence of screen shapes the
flow passed through, `drifted: true|false`, or `null` when the run stopped before every screen
was seen — a guess would be worse than an honest "not judged".

**An unexpected confirmation dialog is handled as a recovery, not a step.** The transfer
capability was recorded against a small amount, so its recording contains no dialog. High-value
transfers raise an extra acknowledgement the flow never saw at record time; replay meets it as a
runtime interruption, matches the app-level `HIGH_VALUE_CONFIRM` recovery, acknowledges it, and
re-proves the step — the same machinery as the maintenance banner. That is the payoff of
declarative, app-level recoveries: a new capability inherits them for free.

**Recovery policy is per rule, and `then` matters.** A maintenance banner is `retry_step`. A
dropped session is `restart_flow`, because the result set the blocked step expected died with the
session. Restart is refused once an irreversible step has executed: replaying a transfer because
the session dropped afterwards is worse than failing.

**Sign-on is an app-level concern, not a step.** Recorded into each capability it makes the
capability unrepeatable when a session already exists and bakes one institution's credentials into
a shared document. It lives once, in the app profile; the same `SIGNED_OUT` rule is the preflight
before a run and the handler mid-flow.

Discovery stops on max steps, a wall-clock timeout, a detected dead-end (repeated action or
unchanging screen), or a model refusal — each a distinct, reported stop reason.

## 4. Heterogeneity & multi-tenant

**Other surfaces.** A UIA provider fills the same `UiElement` from `ControlType`, `Name`,
`LabeledBy` and `BoundingRectangle`; macOS AX from `AXRole`/`AXTitle`/`AXFrame`. The schema does
not change — `frame` becomes a window/pane path, `role` takes the platform's vocabulary — and the
engine, checkpoints, gate and escalation are untouched; the CDP provider is the evidence. What
*would* need work is acting (a native dropdown is not a `<select>`), which is why acting lives
behind the interface while finding does not.

**Many tenants, one product.** Artifacts are keyed to the **vendor product**, never a tenant.
Specialization is a separate `TenantOverlay` that may retarget a control by `semanticId`, remap
screen wording in checkpoints, add tenant-only recoveries, and bind the tenant's origin — and may
**not** add, remove or reorder steps, change the success condition, or widen the allowlist. If
overlays could change the flow, a year later there are N forks nobody can review or upgrade
together. A tenant that runs a genuinely different flow is a different capability version.

Demonstrated, not described: the capability recorded against Northgate runs unchanged against
Vale Mutual — different palette, fonts, reordered result columns — with a **three-line overlay**:
two renamed controls and one reworded title. Steps, checkpoints, outcomes, recoveries and
extraction are reused.

**Drift detection**, cheapest first. Run the unspecialized artifact against a new tenant: the
failure names the exact literal that moved and points at the overlay. The surface fingerprint in
every result flags a capability for re-review when the screens it passed through changed shape —
and names the step. This fired for real during the project: the balance capability was recorded
before a "Transfer Funds" link was added to the member screen, and every replay since reports
`/member:5 → 6` on exactly the steps that touch that screen, through both perception providers
identically. The capability still succeeds; the reviewer knows precisely where to look. That is the
release-shipped event the detector exists for, caught without anyone injecting it.
And the locator rung in every trace is the leading indicator: a fleet-wide slide from `role_name`
to `ordinal` on one capability means that product shipped a release, before anything breaks.

## 5. Escalation & handoff

**Detecting stuck** is four explicit triggers, not one heuristic: the risk ceiling
(`ConfirmationRequired` from the gate); a target with `ambiguity: escalate` resolving to more than
one element; a recovery rule exhausting its budget with `escalateOnExhaustion`; and, during
discovery, the dead-end detector. Plain locator misses and postcondition failures do **not** page
a human — a broken artifact is an engineering problem, and routing every miss to an operator queue
is how the queue gets ignored.

**The structural requirement is session lifetime.** If the runner owned the browser, "take
control of the live session" would be impossible — the window dies with the stuck process. So the
browser is its own process and every run attaches over CDP; automation exits while the session,
cookies and half-filled form stay put.

**Control is an explicit fact.** A single-writer lease records who holds the session. The gate
checks it before *every* action, so a person taking over mid-step is not fighting an automation
still clicking. Control returns with a new token, so a killed-and-restarted runner cannot
resurrect itself into a session someone is now using.

**The loop.** Escalate → file an `InterventionRequest` (capability, step, reason, redacted screen
summary, screenshot, observation snapshot, and **the CDP endpoint of the live session**) → lease
flips to `human` → the operator works in the same window → they **authorize one execution** or
say they did it themselves → lease returns → automation **re-proves the step's precondition**
before acting. If the operator left the session elsewhere, that is a precondition failure, not a
click on whatever now occupies those coordinates.

Authorization is single-use and step-scoped: approving "open this sub-account now" is not
approving every one for the rest of the run. While the human holds the lease, page listeners
record what they clicked, changed and submitted into the same evidence stream — an intervention is
evidence of a gap, and it is the most informative event the system produces.

**Mocked:** the console's presentation — one page, no video, no auth. **Real:** the request and
its context, the lease, the transfer, the blocking of automation, the capture of the human's
actions, single-use authorization, and the verified resume. `npm run demo:handoff` runs the loop
end to end; with `--deposit 10` the human-approved step is then rejected by the app and the run
returns `DEPOSIT_BELOW_MINIMUM` — approval is not the same as success.

## 6. Safety

**Allowlist at one chokepoint.** Origins, path prefixes, action verbs and a maximum unattended
risk class, enforced in the gate both paths act through. The URL is re-checked *after* every
action, because a click can navigate and an allowlist that only guards `navigate` is escaped by
any link. Origins are a tenant binding, supplied by the overlay — a reviewed document — not a
runtime flag.

**Risk is a property of the step, not the verb — and of commitment, not nouns.** Clicking
"Search" and clicking "Confirm Transfer" are the same verb. The recorder proposes a class from the
control's label and writes it into the artifact where a reviewer can raise it; policy decides what
each class may do. The classifier keys on *committing* verbs (confirm, submit, post, authorize,
open/close account), deliberately not on nouns: matching "transfer" made three steps of the
transfer flow irreversible — opening the form, reviewing, confirming — when only the last moves
money. Over-caution that escalates navigation trains operators to wave escalations through, which
is its own safety failure. Above the ceiling the default is to **route to a human, not block**; irreversible steps are
never retried.

**Redaction happens on the read path.** The model never sees a balance or account number; it sees
`«currency:25fe2a»`. A system that shows the model a balance and then scrubs the log has still
disclosed it. Tokens are stable within a run and reversible **in process only**, so the runtime
can type a value back into the page and return the real number to the caller while the evidence
keeps the token.

**Declared sensitivity outranks pattern matching.** A regex cannot recognise a balance normalized
to `18204.37`, and nothing about "J. Whitfield" says regulated. Outputs the artifact declares
`regulated` are masked in the replay log by *declaration*, and once a discovery run has declared
its outputs, every value that ever appeared under one of their labels — including on screens the
model only explored — is scrubbed from the transcript and recording. Labels themselves are never
scrubbed, because the recording must still recompile. A reviewer can raise a sensitivity the model
under-declared (`--raise`), never lower one, and the decision is written into provenance. All three
were live leaks found by grepping the evidence directory — the third being account numbers glued to
a label in nested-table row text (`To999888777`), which a `\b`-anchored pattern silently skips;
pattern redaction is the backstop, typed sensitivity is the control.

**Artifacts are guarded twice.** The compiler will not build a locator or checkpoint from a
literal that looks regulated, equals a caller-supplied value, or is a redaction token, and
`assertNoRegulatedData` re-scans the finished document and fails the build. It caught the compiler
emitting `text: "$18,204.37"` as a locator during development.

**The model can decline.** The first transfer discovery ended in `stop_reason: refusal` — a funds
transfer with no context read as potential fraud. The loop treats that as a distinct
`model_declined` stop reason, and the fix was truthful context in the prompt (a sandboxed
evaluation with synthetic data, an authorized operator task), not a workaround. A system that drives
financial UIs with a model has to expect this and surface it, not retry it.

**Limits.** The model still *sees* a member's name during discovery — the declaration scrub
protects what is persisted, not what was sent — and only a classifier, not a regex, would stop
that. Regex misses a balance rendered as an image, and the currency threshold ($1,000, so "minimum
$25.00" survives) is a configuration choice. Risk classification
from labels is a weak signal, which is why it is reviewable rather than load-bearing. The
allowlist trusts the URL the browser reports. And nothing here defends against a malicious
*target*: a page that renders instructions to the agent is a prompt-injection surface this system
does not address.

## 7. Cuts

**Not built, on purpose.** Queues, workers, a tenant registry, any distribution — the brief
penalizes it. A desktop provider: the seam is argued in §4 and demonstrated with the CDP provider,
but no UIA code exists. A co-browsing console. Self-healing locators — when targeting degrades the
system says so rather than letting a model rewrite the artifact in production. Generalizing from a
single recording. Auth on the operator console (localhost only).

**Next, in order.** (1) Diff the same flow recorded against two tenants: differing literals are
tenant-specific and belong in overlays automatically. (2) A registry keyed on the surface
fingerprint, so degradation across tenants is a query, not an incident. (3) Promote repeated
interventions into recovery rules — the operator's actions are already in the evidence stream.
(4) A UIA provider against a real desktop app, the honest test of §4.
