"""
VERA-style question bank (VERA, CVPR 2025: questions about OBSERVABLE evidence instead of one vague
"is this anomalous?"). Every question is multiple choice with three answers:

    A. Yes — clearly visible      B. No — the frames show otherwise      C. Unclear — not enough visual evidence

and is read as answer-letter probabilities by SmolVLM2 (no free text, no self-reported confidence).
Questions are grouped by category and attached ONLY to the candidate events they verify (a safe event never
triggers customer questions). Versioned: change QUESTION_VERSION when the wording changes (cache key).

Review of the 14 legacy per-frame questions (Qwen engine, pipeline/vlm.py FACTS):
    person_visible, cashier_at_desk, client_at_desk     -> replaced by Grounding DINO people + zones (cheaper)
    outsider_behind_counter                             -> zones + q_outsider (temporal, short clip)
    collector_present, collection_bag_visible           -> q_collector_handover / q_bag_* on the visit clip
    cash_on_desk, drink_over_cash, personal_item_on_desk-> detector states + q_cash_after_leave / q_cup / q_item
    cash_in_hands                                        -> needs a clip, not one frame: q_counting
    safe_open, key_in_safe_lock, service_door_open,     -> single-frame answers were near chance on full frames;
    computer_screen_on                                     now asked on enlarged CROPS of the region found
    (removed: redundant person_visible; ambiguous "client vs collector" merged into the visit questions)
"""
from __future__ import annotations

QUESTION_VERSION = "q-v1"
ANSWERS = ["Yes, it is clearly visible", "No, the frames show otherwise", "Unclear, not enough visual evidence"]

# category -> id -> text.  {obj} etc. are filled by the rule that asks.
BANK = {
    "object_presence": {
        "q_cash_on_desk": "Is cash (banknotes) visibly lying on the desk in these frames?",
        "q_cup": "Is there a cup or a drink on the cash desk next to or above banknotes?",
        "q_item": "Is a personal item (a handbag, backpack, clothing) lying on or right next to the cash desk "
                  "(not a sealed cash-collection bag)?",
        "q_bag_left": "Is a cash-collection bag lying in the room with nobody attending it?",
        "q_mixed_stack": "Does one stack of banknotes contain banknotes of clearly different denominations mixed "
                         "together?",
    },
    "person_role": {
        "q_outsider": "Is a person who is not the cashier (and not a cash collector in uniform) on the cashier's "
                      "side of the counter?",
        "q_workplace_empty": "Is the cashier's workplace empty (the cashier has left)?",
    },
    "possession_interaction": {
        "q_counting": "After the client has left, is the cashier counting banknotes?",
        "q_partial_payout": "Does the cashier hand the client only part of the counted cash and keep the rest?",
        "q_no_document": "Does the cashier accept cash from the client without the client showing an identity "
                         "document?",
        "q_transfer": "Does the sequence visibly show an object moving from one person's hands to another "
                      "person's hands?",
    },
    "temporal_change": {
        "q_cash_after_leave": "The employee has left the workstation. In the frames after that, does the same cash "
                              "remain lying on the desk with nobody attending it?",
        "q_cash_before_client": "A client has just come to the counter. Was cash already lying on the desk before "
                                "the client arrived (left over from before)?",
        "q_count_interrupted": "Does the cashier stop counting the cash in the middle (leaves or does something "
                               "else) and continue later?",
        "q_collector_handover": "Does a cash collector hand over or take a bag and leave quickly, without the cashier "
                                "checking the bag, its seal and the document?",
    },
    "region_state": {
        "q_safe_open": "This is a close-up of the safe (metal cabinet). Is the safe door open?",
        "q_key_in_lock": "This is a close-up of the safe. Is a key inserted in the safe's lock?",
        "q_door_open": "This is a close-up of the room door. Is the door open?",
        "q_screen_on": "This is a close-up of the computer. Is the screen switched on and showing a program "
                       "(not black, not a lock screen)?",
    },
    "event_verification": {
        "q_fallback": "Compared with normal cash-desk work, does this clip show cash or valuables left unattended, "
                      "an open safe with nobody there, or a stranger behind the counter?",
    },
    "contradiction": {
        "q_cash_put_away": "Before leaving, did the employee put the cash away (into a drawer, the safe or a bag)?",
        "q_is_collector": "Is the person a cash collector (uniform, carrying a sealed bag)?",
        "q_someone_watching": "Is another employee at the desk watching the cash?",
    },
}
TEXT = {qid: q for cat in BANK.values() for qid, q in cat.items()}
CATEGORY = {qid: cat for cat, qs in BANK.items() for qid in qs}


def text(qid: str) -> str:
    return TEXT[qid]


def spec(qid: str, role: str, times: list[float], crop=None) -> dict:
    """role: mandatory (must be Yes), support (raises the score), contradiction (Yes lowers / rejects)."""
    assert qid in TEXT, qid
    assert role in ("mandatory", "support", "contradiction")
    return {"id": qid, "role": role, "times": [round(float(t), 2) for t in times], "crop": crop,
            "category": CATEGORY[qid]}
