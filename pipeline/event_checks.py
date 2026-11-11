"""Observable prerequisites. No video names, label times or ground-truth inputs.
These checks operationalize the team's catalog; they do not validate its legal citations.
"""
CHECKS = {
 'safe_key_unattended': {
  'key': 'A distinguishable physical key remains inserted in the safe lock; not just a handle or reflection.',
  'away': 'The responsible cashier moves away from that safe/workplace.',
  'after': 'The key is still visible in the lock AFTER the cashier moves away.'},
 'personal_items_cash_area': {
  'item': 'A recognizable personal handbag, backpack, clothing or other clearly personal item is visible. Standard phones, cash machines, documents and collection bags are not sufficient.',
  'location': 'It is stored on/at the cashier work surface or inside the cash workspace, not merely carried through or held by a visitor.'},
 'cash_unattended': {
  'cash': 'Banknotes are visually distinguishable, not inferred from paper, receipts, a machine, or the banking setting.',
  'away': 'The cashier leaves the working area and no other person visibly takes responsibility there.',
  'after': 'The same banknotes remain exposed on the desk after departure; they were not carried to the safe.'},
 'safe_left_open': {
  'open': 'The safe door is visibly open, not inferred from an unclear angle.',
  'valuables': 'Cash or valuables are visible inside the safe.',
  'away': 'The cashier leaves the safe unattended after using it; briefly opening it while standing beside it is normal.'},
 'unlocked_bank_terminal': {
  'away': 'The cashier leaves the workstation.',
  'unlocked': 'After departure the display visibly shows an active application/session rather than a lock screen. A lit screen or open laptop lid alone does not establish this.'},
 'service_door_left_open': {
  'door': 'The service entrance is identified from the visible room layout; distinguish it from the safe and glass partitions.',
  'open': 'A visible gap or door-leaf position shows that this door is open.',
  'persistent': 'It remains open after someone passes through or during settled work, across at least two distinct timestamps. Brief opening while passing is normal.'},
 'counting_without_customer': {
  'receipt': 'Earlier visible context shows this cashier receiving these banknotes from a customer.',
  'departure': 'That customer subsequently leaves before the count is complete.',
  'counting': 'The cashier continues counting the same received banknotes after that departure. Counting alone without the earlier transaction is insufficient.'},
 'cash_count_interrupted': {
  'counting': 'The cashier is visibly counting a first customer\'s cash.',
  'second': 'A different customer is then served while the first transaction is unfinished.',
  'unfinished': 'The first customer\'s banknotes remain aside unfinished during the switch; an ordinary pause is insufficient.'},
 'cash_collection_without_checks': {
  'handover': 'A cash collection handover is established by visible actions and items, not clothing alone.',
  'coverage': 'The supplied context covers receipt through acceptance/departure with the relevant actions visible.',
  'omission': 'The cashier accepts/puts away the received valuables and the collector leaves with no required visible checking action during that complete sequence.'},
 'collection_bag_not_inspected': {
  'bag': 'A collection bag, distinguishable from a personal handbag, is handed to the cashier.',
  'coverage': 'Receipt through acceptance and collector departure is visible; checks could not simply have happened before the supplied context.',
  'omission': 'The cashier puts the bag aside and the collector leaves without visible inspection of bag/seal/accompanying document.'},
 'collection_bag_unsecured': {
  'bag': 'An identified collection bag with valuables is present, not an ordinary personal handbag.',
  'unsecured': 'It remains in an exposed location rather than secure storage.',
  'unattended': 'The cashier leaves it unattended; holding or processing it normally is insufficient.'},
 'previous_cash_on_table': {
  'previous': 'An earlier customer transaction and its banknotes are visually established.',
  'next': 'A subsequent different customer is being served.',
  'remaining': 'The earlier customer\'s same banknotes remain exposed during the new operation.'},
 'incomplete_cash_payout': {
  'payout': 'A payout and the portion intended for the customer are visibly established.',
  'retained': 'The cashier hands over only part and retains a clearly identifiable intended remainder.',
  'departure': 'The customer leaves without that remainder; do not infer amounts or intent from an arbitrary stack.'},
 'acceptance_without_documents': {
  'receipt': 'A customer hands cash to the cashier.',
  'coverage': 'The visible sequence covers the transaction from receipt through completion with document handling visible.',
  'omission': 'The transaction visibly finishes without the required observable document handling; an unreadable screen or missing close-up is not evidence of missing electronic paperwork.'},
 'unauthorized_cash_area_entry': {
  'visitor': 'Prior visible activity establishes a visitor/customer role without identifying the face.',
  'boundary': 'The service/customer boundary is visually identifiable.',
  'entry': 'That visitor crosses into the cashier-only workspace; approaching the normal customer chair is insufficient.'},
 'mixed_denominations_stack': {
  'denominations': 'Different banknote denominations are visually distinguishable; color alone is insufficient if unreadable.',
  'counted': 'The sequence shows these as already counted/sorted banknotes.',
  'mixed': 'The cashier combines those different denominations into one final stack rather than temporarily sorting.'},
 'coffee_over_cash': {
  'cash': 'Recognizable banknotes are exposed on the work surface.',
  'drink': 'A drink cup is visibly held or used directly above those banknotes; a cup elsewhere is insufficient.'},
}
OBJECT_TYPES = {
 'key': ['safe_key_unattended'],
 'bag': ['personal_items_cash_area', 'collection_bag_unsecured', 'collection_bag_not_inspected'],
 'cash': ['cash_unattended', 'previous_cash_on_table'],
 'cup': ['coffee_over_cash'],
}
GLOBAL_CONTEXT = (
 'Inspect visible actions, object states and transitions, not people\'s identities. '
 'A cashier may normally count, sort, use a counting machine or move cash to a safe alone. '
 'Do not infer money from paper, documents or machines. A safe handle is not a key. '
 'Distinguish personal handbags from collection bags. '
 'Temporary door/safe opening while attended is normal. '
 'A glowing screen is not proof of an unlocked session. '
 'Not seeing a check in a short clip does not prove it was skipped. '
 'Missing or occluded evidence is UNKNOWN, not YES. '
 'No face recognition, no assumptions about accounts, authorization or exact money amounts.'
)
