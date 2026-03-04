from collections import defaultdict, deque

HISTORY = defaultdict(lambda: deque(maxlen=20))

def add_message(chat_id: str, role: str, content: str):
    HISTORY[chat_id].append({"role": role, "content": content})

def get_history(chat_id: str):
    return list(HISTORY[chat_id])

def history_to_text(history, max_turns=6):
    h = history[-max_turns:]
    lines = []
    for m in h:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)