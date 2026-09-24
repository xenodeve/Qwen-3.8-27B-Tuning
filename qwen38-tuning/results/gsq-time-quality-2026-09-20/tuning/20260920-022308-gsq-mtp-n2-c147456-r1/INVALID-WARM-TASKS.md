# Invalid warm task rows

The original-session-replay row is a valid pilot response. The subsequent
session-merge_intervals and session-toposort rows/partial streams are invalid
for task-time or quality ranking: replay_cases replaced the converted trailing
ambient system note instead of the actual user question. Both the original
eight-sentence summary request and the new code request were consequently sent.

The failure was reproduced by
test_warm_task_replaces_real_user_ask_before_trailing_ambient_system_note and
fixed by replacing the last real user question in the Anthropic capture before
conversion, preserving the trailing ambient note. The run was cancelled after
the fault was identified. Preserve all raw output; do not count these warm rows
as evidence of model failure or slow reasoning on the intended code-only task.
