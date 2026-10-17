"""Tái lập rút gọn cơ chế của EvoGraph-R1 (không huấn luyện GRPO).

Tác tử (LLM qua API tương thích OpenAI) tương tác với một KG bị làm khuyết/nhiễu
qua các hành động GRAPH_RETRIEVE / WEB_SEARCH / GRAPH_EDIT / ANSWER.
Mỗi cấu hình (variant) bật một tập hành động khác nhau, log lên Weights & Biases.
"""
import argparse, json, os, random, re, string, time
from collections import defaultdict

import wandb
from openai import OpenAI

VARIANTS = {
    "static": {"GRAPH_RETRIEVE", "ANSWER"},
    "web": {"GRAPH_RETRIEVE", "WEB_SEARCH", "ANSWER"},
    "edit": {"GRAPH_RETRIEVE", "GRAPH_EDIT", "ANSWER"},
    "full": {"GRAPH_RETRIEVE", "WEB_SEARCH", "GRAPH_EDIT", "ANSWER"},
}

SYSTEM = """You answer multi-hop questions about movies using a knowledge graph (KG) of facts (head | relation | tail).
The KG may be INCOMPLETE or contain WRONG facts. Each turn output exactly ONE JSON object (no other text) with an action:
{ACTIONS}
Rules: explore step by step; verify facts; the final answer is a list of entity names. Max {T} turns."""

DOC = {
    "GRAPH_RETRIEVE": '{"action":"GRAPH_RETRIEVE","entity":"<entity name>"}  -> facts touching the entity in the current KG',
    "WEB_SEARCH": '{"action":"WEB_SEARCH","entity":"<entity name>"}  -> facts about the entity found on the web (may be noisy)',
    "GRAPH_EDIT": '{"action":"GRAPH_EDIT","op":"insert"|"delete","fact":["head","relation","tail"]}  -> modify the current KG',
    "ANSWER": '{"action":"ANSWER","answer":["name1","name2"]}  -> stop and answer',
}


def norm(s):
    s = s.lower().strip()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def f1(pred, gold):
    p, g = {norm(x) for x in pred}, {norm(x) for x in gold}
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def load_kb(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                out.append(tuple(parts))
    return out


def load_qa(path, n, seed):
    qa = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, a = line.rstrip("\n").split("\t")
            m = re.search(r"\[(.+?)\]", q)
            if m:
                qa.append((q.replace("[", "").replace("]", ""), m.group(1), a.split("|")))
    random.Random(seed).shuffle(qa)
    return qa[:n]


class KG:
    """KG đầy đủ + lớp phủ (overlay) riêng cho từng câu hỏi: xóa/chèn không ảnh hưởng câu khác."""

    def __init__(self, triples):
        self.facts = set(triples)
        self.by_ent = defaultdict(set)
        for t in triples:
            self.by_ent[t[0]].add(t)
            self.by_ent[t[2]].add(t)
        self.entities = list(self.by_ent)

    def two_hop_paths(self, topic, answers):
        ans, paths = set(answers), []
        for t1 in self.by_ent.get(topic, ()):
            mid = t1[2] if t1[0] == topic else t1[0]
            for t2 in self.by_ent.get(mid, ()):
                end = t2[2] if t2[0] == mid else t2[0]
                if end in ans and t2 != t1:
                    paths.append((t1, t2))
        return paths


class Env:
    def __init__(self, full, topic, answers, args, rng):
        self.full, self.rng, self.args = full, rng, args
        self.removed, self.added = set(), set()
        self.edits = 0
        self.corrupted = False
        paths = full.two_hop_paths(topic, answers)
        if paths and rng.random() < args.drop:  # làm khuyết: xóa một cạnh trên đường suy luận đúng
            self.removed.add(rng.choice(rng.choice(paths)))
            self.corrupted = True
        if paths and rng.random() < args.wrong:  # làm nhiễu: thêm cạnh sai
            h, r, t = rng.choice(paths)[0]
            self.added.add((h, r, rng.choice(full.entities)))
            self.corrupted = True

    def view(self, ent):
        fs = (self.full.by_ent.get(ent, set()) - self.removed) | {f for f in self.added if ent in (f[0], f[2])}
        return sorted(fs)

    def web(self, ent):
        out = []
        for f in sorted(self.full.by_ent.get(ent, set())):
            if self.rng.random() > self.args.web_cov:
                continue
            if self.rng.random() < self.args.web_noise:
                f = (f[0], f[1], self.rng.choice(self.full.entities))
            out.append(f)
        return out[: self.args.k]

    def edit(self, op, fact):
        fact = tuple(fact)
        if len(fact) != 3:
            return "invalid fact"
        self.edits += 1
        if op == "insert":
            self.added.add(fact)
            self.removed.discard(fact)
        elif op == "delete":
            self.removed.add(fact)
            self.added.discard(fact)
        else:
            return "invalid op"
        return "ok"


def fmt(facts, k):
    if not facts:
        return "(no facts found)"
    return "\n".join(" | ".join(f) for f in facts[:k])


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_one(client, args, allowed, env, question):
    sys_p = SYSTEM.replace("{ACTIONS}", "\n".join(DOC[a] for a in sorted(allowed))).replace("{T}", str(args.T))
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": question}]
    counts = defaultdict(int)
    for step in range(args.T):
        for attempt in range(6):  # thử lại khi bị giới hạn tốc độ (429) hoặc lỗi mạng
            try:
                r = client.chat.completions.create(model=args.model, messages=msgs, temperature=0, max_tokens=300)
                break
            except Exception as e:
                fatal = getattr(e, "status_code", None) in (400, 401, 402, 403, 404)
                if attempt == 5 or fatal:
                    raise
                time.sleep(2 ** attempt * 2)
        txt = r.choices[0].message.content or ""
        msgs.append({"role": "assistant", "content": txt})
        a = parse(txt)
        if not a or a.get("action") not in allowed:
            msgs.append({"role": "user", "content": "Invalid action. Output one valid JSON action."})
            continue
        act = a["action"]
        counts[act] += 1
        if act == "ANSWER":
            ans = a.get("answer", [])
            return ([ans] if isinstance(ans, str) else ans), step + 1, counts
        if act == "GRAPH_RETRIEVE":
            obs = fmt(env.view(a.get("entity", "")), args.k)
        elif act == "WEB_SEARCH":
            obs = fmt(env.web(a.get("entity", "")), args.k)
        else:
            obs = env.edit(a.get("op"), a.get("fact", []))
        msgs.append({"role": "user", "content": "Observation:\n" + obs})
    return [], args.T, counts  # hết lượt mà chưa trả lời


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="MetaQA kb.txt (head|relation|tail)")
    ap.add_argument("--qa", required=True, help="MetaQA 2-hop qa_test.txt")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=int, default=8, help="số lượt tối đa")
    ap.add_argument("--k", type=int, default=15, help="số fact tối đa mỗi quan sát")
    ap.add_argument("--drop", type=float, default=0.5, help="xác suất xóa 1 cạnh trên đường đúng")
    ap.add_argument("--wrong", type=float, default=0.2, help="xác suất chèn 1 cạnh sai")
    ap.add_argument("--web_cov", type=float, default=0.8, help="xác suất một fact có trên 'web'")
    ap.add_argument("--web_noise", type=float, default=0.1, help="xác suất fact trên web bị sai")
    ap.add_argument("--model", default="qwen/qwen3-32b")
    ap.add_argument("--base_url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--api_key_env", default="GROQ_API_KEY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default="evograph-reproduction")
    ap.add_argument("--wandb_group", default="main-ablation")
    args = ap.parse_args()

    run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                     name=f"{args.variant}-seed{args.seed}", config=vars(args))
    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env])
    full = KG(load_kb(args.kb))
    qa = load_qa(args.qa, args.n, args.seed)
    allowed = VARIANTS[args.variant]
    tab = wandb.Table(columns=["question", "gold", "pred", "f1", "steps", "corrupted", "edits", "actions"])
    hits = f1s = steps = edits = 0.0
    c_hit = [0, 0]  # [đúng, tổng] trên câu bị làm khuyết/nhiễu
    n_c = 0
    for i, (q, topic, gold) in enumerate(qa):
        env = Env(full, topic, gold, args, random.Random(args.seed * 100003 + i))  # cùng nhiễu cho mọi variant
        try:
            pred, st, cnt = run_one(client, args, allowed, env, q)
        except Exception as e:  # lỗi API
            if getattr(e, "status_code", None) in (400, 401, 402, 403, 404):
                run.finish(exit_code=1)
                raise
            print("API error:", e)
            time.sleep(5)
            pred, st, cnt = [], args.T, {}
        s = f1(pred, gold)
        hit = float(any(norm(p) in {norm(g) for g in gold} for p in pred))
        hits += hit; f1s += s; steps += st; edits += env.edits
        if env.corrupted:
            n_c += 1; c_hit[0] += hit; c_hit[1] += 1
        tab.add_data(q, "|".join(gold), "|".join(pred), s, st, env.corrupted, env.edits, json.dumps(dict(cnt)))
        n = i + 1
        wandb.log({"step": n, "hit_running": hits / n, "f1_running": f1s / n})
    n = len(qa)
    wandb.log({"hit": hits / n, "f1": f1s / n, "avg_turns": steps / n, "avg_edits": edits / n,
               "hit_on_corrupted": c_hit[0] / max(1, c_hit[1]), "n_corrupted": n_c, "n": n, "samples": tab})
    print(f"[{args.variant}] hit={hits/n:.3f} f1={f1s/n:.3f} turns={steps/n:.2f} edits={edits/n:.2f}")
    run.finish()


if __name__ == "__main__":
    main()
