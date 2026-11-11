# Violation rules (verified engine)

Types and legal references come from `catalog/violations.yaml` (unchanged). The output is a **visual event
suspected to violate a rule**; the legal assessment is left to a person (every description says so).
References marked `conditional`, `related_not_direct` or `no_direct_clause` in the catalog need human
confirmation. Durations are in `configs/verified.yaml → rules`.

| Violation | Required sequence (detector + time) | VLM verification (role) | Contradiction / stops the rule | Max status |
|---|---|---|---|---|
| cash_unattended | cashier present → cash on desk at departure → cashier absent ≥ 3 s → the same cash stays ≥ 3 s | q_cash_after_leave (support) | cash removed before leaving (desk visible, no cash) stops it; q_cash_put_away lowers it; cash UNKNOWN while away → review | confirmed |
| previous_cash_on_table | cash on the desk for ≥ 3 s BEFORE a client arrives | q_cash_before_client (support) | any visible empty desk in that window stops it | confirmed |
| coffee_over_cash | cup and cash boxes within 1.5 cup sizes for ≥ 2 s | q_cup (support) | — | confirmed |
| personal_items_cash_area | bag / handbag / backpack in the desk zone ≥ 6 s (a phone is normal per the catalog) | q_item on the object crop (support) | q_is_collector lowers it | confirmed |
| unauthorized_cash_area_entry | a second person inside the cashier zone ≥ 2 s | q_outsider (mandatory) | q_is_collector lowers it | confirmed |
| collection_bag_not_inspected / cash_collection_without_checks | visit with a bag; visit ≤ 8 s | q_collector_handover (mandatory) | longer visit → condition UNKNOWN | confirmed |
| collection_bag_unsecured | bag still there after the visitor left, nobody at the desk ≥ 3 s | q_bag_left (support) | bag gone (visible) stops it | confirmed |
| safe_left_open | cashier absent ≥ 3 s | q_safe_open on the safe close-up (mandatory) | VLM "No" → rejected | confirmed |
| safe_key_unattended | cashier absent ≥ 3 s (+ key box near the safe if detected) | q_key_in_lock on the safe close-up (mandatory) | VLM "No" → rejected | confirmed |
| unlocked_bank_terminal | cashier absent ≥ 5 s, screen region found | q_screen_on on the screen close-up (mandatory) | VLM "No" → rejected | confirmed |
| service_door_left_open | people present; door region found; sampled every 4 s (≤ 12 samples) | q_door_open per sample, must be "Yes" on consecutive samples ≥ 6 s | all "No" → rejected | confirmed |
| counting_without_customer | client leaves, cashier at desk within 10 s | q_counting (mandatory) | client back → no candidate | confirmed |
| incomplete_cash_payout, acceptance_without_documents, cash_count_interrupted, mixed_denominations_stack | client visit ≥ 4 s (amounts, documents, denominations are not readable by the detector) | one targeted question each (mandatory) | — | **needs review only** |
| key_transfer (internal, not a label) | key next to person A ≥ 2 frames, then next to person B ≥ 2 frames | q_transfer (mandatory) | no visual evidence → rejected | internal |
| other (fallback) | detector found no departure and no visit at all: 6 periodic windows | q_fallback (mandatory) | — | needs review only |

Score = rule_score × Π_mandatory(0.3 + 0.7·s) × (0.5 + 0.5·mean support) × contradiction × (0.6 if any
condition UNKNOWN), with s = p(yes) + ½ p(unclear). A FALSE detector condition never produces a candidate,
so no VLM answer can override it.
