# app.py
"""
Predicate Logic Resolution Prover (Streamlit single-file app)

Usage:
  1. Create virtual env (recommended): python -m venv .venv
  2. Activate it: (Windows) .venv\\Scripts\\activate   (mac/linux) source .venv/bin/activate
  3. Install: pip install -r requirements.txt
  4. Download spaCy model: python -m spacy download en_core_web_sm
  5. Run: streamlit run app.py

This app provides:
 - English -> predicate conversion (rule-based + spaCy fallback)
 - CNF conversion pipeline (implication elimination, NNF, distribute OR over AND)
 - Clause extraction
 - A basic first-order resolution prover (unification + resolution loop)
 - Interactive UI with step-by-step display for presentation
"""

import streamlit as st
import re
import copy
from collections import deque
import os

# Try to import spaCy; provide helpful message if not available
try:
    import spacy
except Exception as e:
    st.error("spaCy is required. Please install it (pip install spacy) and download model en_core_web_sm: python -m spacy download en_core_web_sm")
    raise

# load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # If model is not present, try to download (best-effort)
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ----------------- Page layout & styling -----------------
st.set_page_config(page_title="Predicate Logic Resolution Prover", layout="wide")
st.markdown("""
<style>
    .main { background-color: #f8fafc; color: #0f172a; }
    .title { text-align: center; font-size: 2.2em; color: #0ea5e9; margin-bottom: 0.2em; }
    .subtitle { text-align: center; color: #64748b; margin-bottom: 1.5em; }
    .section-title { font-size: 1.2em; color: #075985; margin-top: 1em; margin-bottom:0.4em; }
    .streamlit-expanderHeader { font-weight: 700; }
    textarea, .stTextInput > div > div > input { font-size: 1.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Predicate Logic Proof Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>English → Predicate Logic → CNF → Resolution (step-by-step)</div>", unsafe_allow_html=True)
st.divider()

# ----------------- FOL internal representation & parser -----------------
# We'll use a small formula representation: ('pred', name, [terms]) / ('not',F) / ('and',F1,F2) / ('or',F1,F2) / ('implies',F1,F2) / ('iff',F1,F2)
# Terms: ('var','x') / ('const','Name') / ('func','f',[...])

TOKEN_SPEC = [
    ('SKIP',   r"[ \t]+"),
    ('LP',     r"\("),
    ('RP',     r"\)"),
    ('COMMA',  r","),
    ('NEG',    r"~|not\b"),
    ('IFF',    r"<=>|iff\b"),
    ('IMPL',   r"=>|implies\b"),
    ('AND',    r"&|and\b"),
    ('OR',     r"\||or\b"),
    ('FORALL', r"forall|∀"),
    ('EXISTS', r"exists|∃"),
    ('DOT',    r"\."),
    ('NAME',   r"[A-Za-z_][A-Za-z0-9_]*"),
    ('NEWLINE', r"\n"),
]



TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)


class Token:
    def __init__(self, typ, val):
        self.type = typ
        self.val = val
    def __repr__(self):
        return f"Token({self.type},{self.val})"

def tokenize(s):
    for mo in re.finditer(TOKEN_REGEX, s):
        typ = mo.lastgroup
        val = mo.group(typ)
        if typ == 'SKIP' or typ == 'NEWLINE':
            continue
        yield Token(typ, val)

class Parser:
    def __init__(self, text):
        self.tokens = deque(tokenize(text))

    def peek(self):
        return self.tokens[0] if self.tokens else None
    def pop(self, expected=None):
        if not self.tokens:
            return None
        t = self.tokens.popleft()
        if expected and t.type != expected:
            raise ValueError(f"Expected {expected}, got {t}")
        return t

    def parse_formula(self):
        return self.parse_iff()

    def parse_iff(self):
        left = self.parse_imp()
        while self.peek() and self.peek().type == 'IFF':
            self.pop('IFF')
            right = self.parse_imp()
            left = ('iff', left, right)
        return left

    def parse_imp(self):
        left = self.parse_or()
        while self.peek() and self.peek().type == 'IMPL':
            self.pop('IMPL')
            right = self.parse_or()
            left = ('implies', left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() and self.peek().type == 'OR':
            self.pop('OR')
            right = self.parse_and()
            left = ('or', left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek() and self.peek().type == 'AND':
            self.pop('AND')
            right = self.parse_not()
            left = ('and', left, right)
        return left

    def parse_not(self):
        if self.peek() and self.peek().type == 'NEG':
            self.pop('NEG')
            operand = self.parse_not()
            return ('not', operand)
        return self.parse_quantifier()
    
    def parse_quantifier(self):
    # Handle quantifiers like forall x. φ or exists x. φ
        if self.peek() and self.peek().type in ('FORALL', 'EXISTS'):
            quant_type = self.pop().type
            vars_ = []
            while self.peek() and self.peek().type == 'NAME':
                vars_.append(self.pop('NAME').val)
                if self.peek() and self.peek().type == 'COMMA':
                    self.pop('COMMA')
                    continue
                break
        # Optional dot or colon after variable(s)
            if self.peek() and self.peek().type == 'DOT':
                self.pop('DOT')
        # Parse full logical formula as quantifier body (not just atom)
            body = self.parse_formula()
        # Wrap for multiple variables (forall x,y,z.)
            for v in reversed(vars_):
                body = (quant_type.lower(), v, body)
            return body
        else:
            return self.parse_atom()


    def parse_atom(self):
        t = self.peek()
        if not t:
            raise ValueError('Unexpected end')
        if t.type == 'LP':
            self.pop('LP')
            f = self.parse_formula()
            self.pop('RP')
            return f
        if t.type == 'NAME':
            name = self.pop('NAME').val
            if self.peek() and self.peek().type == 'LP':
                self.pop('LP')
                args = []
                if self.peek() and self.peek().type != 'RP':
                    while True:
                        arg = self.parse_term()
                        args.append(arg)
                        if self.peek() and self.peek().type == 'COMMA':
                            self.pop('COMMA')
                            continue
                        break
                self.pop('RP')
                return ('pred', name, args)
            else:
                # zero-ary predicate / constant-as-predicate
                return ('pred', name, [])
        raise ValueError(f"Unexpected token {t}")

    def parse_term(self):
        t = self.peek()
        if t.type == 'NAME':
            name = self.pop('NAME').val
            if self.peek() and self.peek().type == 'LP':
                self.pop('LP')
                args = []
                if self.peek() and self.peek().type != 'RP':
                    while True:
                        args.append(self.parse_term())
                        if self.peek() and self.peek().type == 'COMMA':
                            self.pop('COMMA')
                            continue
                        break
                self.pop('RP')
                return ('func', name, args)
            if name[0].islower():
                return ('var', name)
            return ('const', name)
        raise ValueError('Invalid term')

# ----------------- CNF pipeline (simplified) -----------------
var_count = 0
skolem_count = 0

def fresh_var(prefix='v'):
    global var_count
    var_count += 1
    return ('var', f"{prefix}{var_count}")

def fresh_skolem(prefix='sk'):
    global skolem_count
    skolem_count += 1
    return ('func', f"{prefix}{skolem_count}", [])

def substitute_term(term, subs):
    ttype = term[0]
    if ttype == 'var':
        return subs.get(term[1], term)
    if ttype == 'const':
        return term
    if ttype == 'func':
        return ('func', term[1], [substitute_term(a, subs) for a in term[2]])
    raise ValueError('Unknown term type')

def substitute_formula(formula, subs):
    typ = formula[0]
    if typ == 'pred':
        return ('pred', formula[1], [substitute_term(t, subs) for t in formula[2]])
    if typ == 'not':
        return ('not', substitute_formula(formula[1], subs))
    if typ in ('and','or','implies','iff'):
        return (typ, substitute_formula(formula[1], subs), substitute_formula(formula[2], subs))
    raise ValueError('Unknown formula type')

def eliminate_implications(f):
    typ = f[0]
    if typ == 'pred':
        return f
    if typ == 'not':
        return ('not', eliminate_implications(f[1]))
    if typ == 'implies':
        A = eliminate_implications(f[1])
        B = eliminate_implications(f[2])
        return ('or', ('not', A), B)
    if typ == 'iff':
        A = eliminate_implications(f[1])
        B = eliminate_implications(f[2])
        return ('and', ('or', ('not', A), B), ('or', ('not', B), A))
    if typ in ('and','or'):
        return (typ, eliminate_implications(f[1]), eliminate_implications(f[2]))
    raise ValueError('Unknown')

def move_negation(f):
    typ = f[0]
    if typ == 'pred':
        return f
    if typ == 'not':
        inner = f[1]
        if inner[0] == 'pred':
            return f
        if inner[0] == 'not':
            return move_negation(inner[1])
        if inner[0] == 'and':
            return ('or', move_negation(('not', inner[1])), move_negation(('not', inner[2])))
        if inner[0] == 'or':
            return ('and', move_negation(('not', inner[1])), move_negation(('not', inner[2])))
        raise ValueError('Unexpected negation of non-boolean in this simplified pipeline')
    if typ in ('and','or'):
        return (typ, move_negation(f[1]), move_negation(f[2]))
    raise ValueError('Unknown in nnf')

def distribute_or(a, b):
    if a[0] == 'and':
        return ('and', distribute_or(a[1], b), distribute_or(a[2], b))
    if b[0] == 'and':
        return ('and', distribute_or(a, b[1]), distribute_or(a, b[2]))
    return ('or', a, b)

def to_cnf_tree(f):
    typ = f[0]
    if typ in ('pred','not'):
        return f
    if typ == 'and':
        return ('and', to_cnf_tree(f[1]), to_cnf_tree(f[2]))
    if typ == 'or':
        left = to_cnf_tree(f[1])
        right = to_cnf_tree(f[2])
        return distribute_or(left, right)
    raise ValueError('Unexpected type in to_cnf_tree: ' + str(typ))

def flatten_and(f):
    if f[0] == 'and':
        return flatten_and(f[1]) + flatten_and(f[2])
    return [f]

def flatten_or(f):
    if f[0] == 'or':
        return flatten_or(f[1]) + flatten_or(f[2])
    return [f]

def literal_to_tuple(lit):
    if lit[0] == 'not':
        p = lit[1]
        if p[0] != 'pred':
            raise ValueError('Negation of non-predicate in literal')
        return (False, p[1], tuple(p[2]))
    if lit[0] == 'pred':
        return (True, lit[1], tuple(lit[2]))
    raise ValueError('Not a literal')

def term_to_str(t):
    if t[0] == 'var':
        return t[1]
    if t[0] == 'const':
        return t[1]
    if t[0] == 'func':
        return f"{t[1]}({', '.join(term_to_str(a) for a in t[2])})"
    return str(t)

def lit_to_str(l):
    sign, name, args = l
    s = f"{name}({', '.join(term_to_str(a) for a in args)})"
    return ('' if sign else '~') + s

# ----------------- Unification & resolution -----------------
def apply_subs_term(term, subs):
    if term[0] == 'var':
        return subs.get(term[1], term)
    if term[0] == 'const':
        return term
    if term[0] == 'func':
        return ('func', term[1], [apply_subs_term(a, subs) for a in term[2]])
    return term

def occurs_check(vname, term, subs):
    term = apply_subs_term(term, subs)
    if term[0] == 'var':
        return term[1] == vname
    if term[0] == 'const':
        return False
    if term[0] == 'func':
        return any(occurs_check(vname, a, subs) for a in term[2])
    return False

def unify_terms(t1, t2, subs):
    t1 = apply_subs_term(t1, subs)
    t2 = apply_subs_term(t2, subs)
    if t1 == t2:
        return subs
    if t1[0] == 'var':
        if occurs_check(t1[1], t2, subs):
            return None
        new = subs.copy()
        new[t1[1]] = t2
        return new
    if t2[0] == 'var':
        if occurs_check(t2[1], t1, subs):
            return None
        new = subs.copy()
        new[t2[1]] = t1
        return new
    if t1[0] == 'const' and t2[0] == 'const' and t1[1] == t2[1]:
        return subs
    if t1[0] == 'func' and t2[0] == 'func' and t1[1] == t2[1] and len(t1[2]) == len(t2[2]):
        for a,b in zip(t1[2], t2[2]):
            subs = unify_terms(a,b,subs)
            if subs is None:
                return None
        return subs
    return None

def unify_literals(l1, l2):
    if l1[1] != l2[1] or l1[0] == l2[0] or len(l1[2]) != len(l2[2]):
        return None
    subs = {}
    for a,b in zip(l1[2], l2[2]):
        subs = unify_terms(a,b,subs)
        if subs is None:
            return None
    return subs

def normalize_key(k):
    """Ensure all keys are comparable tuples of strings."""
    if not k:
        return tuple()
    if isinstance(k, str):
        return tuple(k.split(" ∨ ")) if " ∨ " in k else (k,)
    if isinstance(k, (list, set, frozenset)):
        return tuple(sorted(k))
    return tuple(k)

def resolution_prove(clauses, query_clause, max_steps=5000):
    """
    Try to prove query_clause using resolution refutation.
    Returns (proved: bool, proof_parents: dict, final_empty_key: tuple)
    """
    kb = [list(c) for c in clauses]
    for l in query_clause:
        kb.append([(not l[0], l[1], l[2])])

    seen = set()
    proof_parents = {}
    steps = 0
    agenda = deque((i, j) for i in range(len(kb)) for j in range(i + 1, len(kb)))

    while agenda and steps < max_steps:
        i, j = agenda.popleft()
        C1, C2 = kb[i], kb[j]

        for l1 in C1:
            for l2 in C2:
                subs = unify_literals(l1, l2)
                if subs is None:
                    continue

                newlits = []
                for lit in C1:
                    if lit is not l1:
                        newlits.append((lit[0], lit[1], tuple(apply_subs_term(t, subs) for t in lit[2])))
                for lit in C2:
                    if lit is not l2:
                        newlits.append((lit[0], lit[1], tuple(apply_subs_term(t, subs) for t in lit[2])))

                newlits_str = tuple(sorted(set(lit_to_str(l) for l in newlits)))

                if newlits_str in seen:
                    continue
                seen.add(newlits_str)

                key_left = tuple(sorted(lit_to_str(l) for l in C1))
                key_right = tuple(sorted(lit_to_str(l) for l in C2))
                proof_parents[normalize_key(newlits_str)] = {
                    "from": (normalize_key(key_left), normalize_key(key_right)),
                    "resolved_on": lit_to_str(l1),
                }

                if len(newlits) == 0:
                    # Record the empty clause in proof map so proof tree can show it
                    key_left = tuple(sorted(lit_to_str(l) for l in C1))
                    key_right = tuple(sorted(lit_to_str(l) for l in C2))
                    proof_parents[normalize_key(())] = {
                        "from": (normalize_key(key_left), normalize_key(key_right)),
                        "resolved_on": lit_to_str(l1),
                    }
                    print("🧩 Proof map keys:", list(proof_parents.keys())[-5:])
                    print("🧩 Final key:", normalize_key(()))
                    return True, proof_parents, normalize_key(())
                
                kb.append(newlits)
                new_index = len(kb) - 1
                for k in range(new_index):
                    agenda.append((k, new_index))
        steps += 1
    return False, proof_parents, None


def build_proof_tree(final_key, proof_map, depth=0, visited=None):
    """
    Recursively reconstruct the resolution proof steps (human-readable).
    More robust version with strict normalization of all keys.
    """
    if visited is None:
        visited = set()

    key = normalize_key(final_key)
    if key in visited:
        return ""
    visited.add(key)

    entry = proof_map.get(key)
    if not entry:
        return ""

    left, right = entry["from"]
    resolved_on = entry["resolved_on"]

    s = ""
    s += build_proof_tree(left, proof_map, depth + 1, visited)
    s += build_proof_tree(right, proof_map, depth + 1, visited)

    indent = "  " * depth
    clause_str = " ∨ ".join(key) if key else "⊥"
    s += f"{indent}{clause_str}   ← resolved on {resolved_on}\n"
    return s

# ----------------- English -> predicate converter (rule-based + spaCy fallback) -----------------
def english_to_predicate(sentence):
    s = sentence.strip().rstrip('.').strip()
    s_lower = s.lower()

    # common, explicit rules for the presentation sentences (covers your 7 statements)
    rules = [
        (r"john owns a dog", "Owns(John, Dog)"),
        (r"mary buys carrots by the bushel", "Buys(Mary, CarrotsByBushel)"),
        (r"anyone who buys carrots by the bushel owns either a rabbit or a grocery store",
         "forall x. (Buys(x, CarrotsByBushel) => Owns(x, Rabbit) | Owns(x, GroceryStore))"),
        (r"every dog chases some rabbit", "forall x. (Dog(x) => exists y. (Rabbit(y) & Chases(x, y)))"),
        (r"anyone who owns a rabbit hates anything that chases any rabbit",
         "forall x. (Owns(x, Rabbit) => forall y. (Chases(y, Rabbit) => Hates(x, y)))"),
        (r"someone who hates something owned by another person will not date that person",
         "forall x,y,z. ((Hates(x, z) & Owns(y, z)) => ~Dates(x, y))"),
        (r"if mary does not own a grocery store, she will not date john",
         "~Owns(Mary, GroceryStore) => ~Dates(Mary, John)"),
    ]
    for pattern, logic in rules:
        if re.search(pattern, s_lower):
            return logic

    # fallback: use spaCy to identify subject-verb-object triplet(s)
    doc = nlp(s)
    subj, verb, obj = None, None, None
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subj = token.text
        elif token.dep_ == "ROOT":
            verb = token.lemma_
        elif token.dep_ in ("dobj", "pobj", "attr"):
            obj = token.text
    if subj and verb and obj:
        # heuristics: map to Pred(Subj, Obj) with capitalized constants
        S = subj.capitalize()
        V = verb.capitalize()
        O = obj.capitalize()
        return f"{V}({S}, {O})"
    return f"# Unable to parse: {sentence}"

# ----------------- UI wiring -----------------
st.markdown("<div class='section-title'>🗣️ Step 1 — Natural Language Input</div>", unsafe_allow_html=True)

example_sentences = """John owns a dog.
Anyone who buys carrots by the bushel owns either a rabbit or a grocery store.
Every dog chases some rabbit.
Mary buys carrots by the bushel.
Anyone who owns a rabbit hates anything that chases any rabbit.
Someone who hates something owned by another person will not date that person.
If Mary does not own a grocery store, she will not date John.
"""
nl_input = st.text_area("Enter English sentences (one per line)", example_sentences, height=200)

convert_col1, convert_col2 = st.columns([1,1])
with convert_col1:
    if st.button("Convert all to Predicate Logic"):
        st.markdown("**Converted predicate logic (best-effort / rule-based):**")
        converted_list = []
        for line in nl_input.splitlines():
            line = line.strip()
            if not line:
                continue
            converted = english_to_predicate(line)
            converted_list.append(converted)
            st.write(f"- **{line}**  →  `{converted}`")
        st.session_state['converted'] = converted_list if converted_list else []
with convert_col2:
    st.info("Tip: Conversion is rule-based + spaCy fallback. For full formal proofs, you may edit the predicate logic lines before running CNF and resolution.")

st.divider()
st.markdown("<div class='section-title'>✍️ Step 2 — Predicate Logic Input (edit / refine)</div>", unsafe_allow_html=True)

# Pre-fill predicate area with converted (if available) or a helpful default
pred_default = "\n".join(st.session_state.get('converted', [
    "Owns(John, Dog)",
    "forall x. (Buys(x, CarrotsByBushel) => Owns(x, Rabbit) | Owns(x, GroceryStore))",
    "forall x. (Dog(x) => exists y. (Rabbit(y) & Chases(x, y)))",
    "Buys(Mary, CarrotsByBushel)",
    "forall x. (Owns(x, Rabbit) => forall y. (Chases(y, Rabbit) => Hates(x, y)))",
    "forall x,y,z. ((Hates(x, z) & Owns(y, z)) => ~Dates(x, y))",
    "~Owns(Mary, GroceryStore) => ~Dates(Mary, John)"
]))

pred_input = st.text_area("Predicate logic lines (one per line). Use: ~ negation, & and, | or, => implies, parentheses, predicates like P(a,b).",
                          value=pred_default, height=240)

# We'll support a simple 'forall/exists' notation in lines by pre-processing them into a form the parser understands:
def preprocess_quantifiers(line):
    # convert `forall x. ( ... )` -> simply return expression inside, assuming variables are implicitly universal
    # convert `exists y. ...` -> keep as marker 'EXISTS_y' for now (we won't implement full quantifier skolemization automatically here)
    # For this demo, encourage users to write formulas already near-clause form or use explicit predicate forms.
    s = line.strip()
    s = re.sub(r"\bforall\s+([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*", "", s, flags=re.I)
    # mark exists for visual but we will treat as part of manual transformation step
    s = re.sub(r"\bexists\s+([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*", "EXISTS_", s, flags=re.I)
    return s

# Parse predicate lines into internal formula representation when requested
parse_errors = []
parsed_formulas = []
lines = [ln.strip() for ln in pred_input.splitlines() if ln.strip() and not ln.strip().startswith('#')]
for i, ln in enumerate(lines):
    try:
        processed = preprocess_quantifiers(ln)
        p = Parser(processed).parse_formula()
        parsed_formulas.append(p)
    except Exception as e:
        parse_errors.append((i+1, str(e), ln))

if parse_errors:
    st.error(f"Parsing error on line {parse_errors[0][0]}: {parse_errors[0][2]}  —  {parse_errors[0][1]}")

st.markdown("<div class='section-title'>🔁 Step 3 — CNF Conversion (select a line to inspect)</div>", unsafe_allow_html=True)
sel_idx = st.number_input("Select formula line number to inspect CNF conversion (1-based)", min_value=1, max_value=max(1,len(parsed_formulas)), value=1)
if parsed_formulas:
    chosen = parsed_formulas[sel_idx-1]
    st.write("Parsed formula tree:")
    st.write(chosen)

    f_noimp = eliminate_implications(chosen)
    st.write("After eliminating implications / iff:")
    st.write(f_noimp)

    try:
        f_nnf = move_negation(f_noimp)
        st.write("After moving negation inward (NNF):")
        st.write(f_nnf)
        cnf_tree = to_cnf_tree(f_nnf)
        st.write("After distribution to CNF tree:")
        st.write(cnf_tree)
        st.write("Extracted clauses from this formula:")
        for cl in flatten_and(cnf_tree):
            lits = [literal_to_tuple(l) for l in flatten_or(cl)]
            st.write([lit_to_str(l) for l in lits])
    except Exception as e:
        st.warning("CNF pipeline encountered an issue for this formula (likely due to EXISTS markers or complex quantifiers). You can edit the predicate input to simpler forms for demonstration.")
        st.write("Error detail:", str(e))

st.divider()
st.markdown("<div class='section-title'>🧩 Step 4 — Full KB Clause Set & Resolution Proof</div>", unsafe_allow_html=True)

# Build global clause set (best-effort)
all_clauses = []
for f in parsed_formulas:
    try:
        fni = eliminate_implications(f)
        fnn = move_negation(fni)
        ctree = to_cnf_tree(fnn)
        for cl in flatten_and(ctree):
            lits = [literal_to_tuple(l) for l in flatten_or(cl)]
            all_clauses.append(lits)
    except Exception:
        pass

st.write(f"Total clauses extracted (best-effort): {len(all_clauses)}")
for i, c in enumerate(all_clauses, 1):
    st.write(f"{i}. " + ", ".join(lit_to_str(l) for l in c))

# Query input
st.markdown("**Enter a query to prove (will be negated for refutation):**")
query_input = st.text_input("Query (e.g. Owns(John, Dog))", value="Chases(d1, f(d1))")

# Parse query
q_clauses = []
try:
    qtree = Parser(query_input).parse_formula()
    q_noimp = eliminate_implications(qtree)
    q_nnf = move_negation(q_noimp)
    q_cnf = to_cnf_tree(q_nnf)
    for cl in flatten_and(q_cnf):
        lits = [literal_to_tuple(l) for l in flatten_or(cl)]
        q_clauses.append(lits)
except Exception as e:
    st.error("Query parse error: " + str(e))
    q_clauses = []

if st.button("Run resolution (refutation)"):
    if not q_clauses:
        st.error("Query parse failed or empty.")
    else:
        proved, proof_map, final_key = resolution_prove(all_clauses, q_clauses[0], max_steps=5000)
        if proved:
            st.success("✅ Query proved by refutation (empty clause derived).")
            st.write("### Proof Steps:")
            proof_str = build_proof_tree(final_key, proof_map)
            st.code(proof_str if proof_str else "(proof trace unavailable)", language="text")
        else:
            st.warning("❌ Could not derive empty clause within step limit.")
            st.write("Partial proof trace (if any):")
            if proof_map:
                for k, v in list(proof_map.items())[:10]:
                    st.write(f"{' ∨ '.join(k) if k else '∅'} ← resolved on {v['resolved_on']}")
            else:
                st.write("(no resolvents generated)")

st.markdown("---")
st.caption("This tool is a pedagogical demo. For robust theorem proving: add explicit quantifier handling & Skolemization, better search heuristics (set-of-support), clause indexing and normalization.")
