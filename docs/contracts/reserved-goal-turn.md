# Host-reserved goal turns

The goal loop advertises `goal.reserved_turn_accounting = 1` at execution entry,
including when inherited by loop-live. A host may expose the synchronous
`goal.turn_already_charged(goal) -> bool` capability. It is consulted only where
the loop would normally increment `turns_used`; resumed goal evaluation retains
its existing accounting path.

Return `true` only when a durable host witness proves this exact goal's current
logical turn has already been charged. The witness must be idempotent after a
lost response or interruption, and distinguish later automatic turns even when
they share the original input and generation. `continuations` advances before
each automatic turn and can be part of this identity. Do not consume the witness
by returning false on a retry. The host owns admission, identity, durable charge,
and budget validation; this capability does not grant authority or replay work.

An absent capability or `false` preserves normal increment behavior. A read or
write failure must raise, rather than claim a reservation is absent. Non-boolean
returns fail visibly. Hosts requiring this accounting must check support before
installing a reserved successor goal. No goal or turn counter is refunded.
