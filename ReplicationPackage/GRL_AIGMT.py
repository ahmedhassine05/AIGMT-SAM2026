#!/usr/bin/env python3
"""
GRL AI Transformer — Three-Phase Pipeline
==========================================
Transforms a TGRL/GRL model file (.xgrl) into an AI-enabled TGRL
specification by running three sequential phases:

  Phase 1 — AI Readiness Assessment
  Phase 2 — AI Transformation Patterns
  Phase 3 — LLM-Based TGRL Transformation

Usage
-----
    python grl_transformer.py model.xgrl

Requires the OPENAI_API_KEY environment variable to be set.
Uses the gpt-5.5 model with default parameters.

Output files (written to the same directory as the input file)
--------------------------------------------------------------
    <prefix>_phase1_readiness.json        Phase 1 full readiness report
    <prefix>_phase2_transformations.json  Phase 2 transformation specifications
    <prefix>_phase3_transformed.xgrl      Phase 3 AI-enabled TGRL model
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

MODEL = 'gpt-5.5'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
# gpt-5.5 is a reasoning model — use max_completion_tokens, skip temperature
P1_MAX_TOKENS = 8000    # per-task readiness assessment
P2_MAX_TOKENS = 8000    # per-task transformation spec
P3_MAX_TOKENS = 32000   # whole-model TGRL rewrite


# ════════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ════════════════════════════════════════════════════════════════════════════════

class LLMError(Exception):
    pass


def _http_post_json(url, payload, api_key, timeout=300):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method='POST',
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise LLMError(f'HTTP {e.code}: {body}')
    except urllib.error.URLError as e:
        raise LLMError(str(e))


def llm_call(system_prompt, user_prompt, api_key, max_tokens):
    """Single OpenAI chat-completions call; returns the assistant content string."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_prompt})
    body = {
        'model': MODEL,
        'messages': messages,
        'max_completion_tokens': max_tokens,
    }
    raw = _http_post_json(OPENAI_API_URL, body, api_key)
    d = json.loads(raw)
    try:
        content = d['choices'][0]['message']['content']
        return content or ''
    except (KeyError, IndexError):
        raise LLMError('Unexpected API response: ' + raw[:500])


def strip_fences(text):
    """Remove markdown code fences if the model wrapped its output in them."""
    t = (text or '').strip()
    m = re.match(r'```(?:json|tgrl|xgrl|grl)?\s*([\s\S]*?)\s*```', t, re.IGNORECASE)
    return m.group(1).strip() if m else t


def extract_json(text):
    """Parse JSON from LLM output, tolerating fences and leading prose."""
    stripped = strip_fences(text)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start, end = stripped.find('{'), stripped.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except Exception:
            pass
    raise ValueError('Response was not valid JSON.')


def safe_base_name(file_name):
    base = re.sub(r'\.xgrl$', '', file_name or 'grl_model', flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', base) or 'grl_model'


# ════════════════════════════════════════════════════════════════════════════════
# .xgrl / TGRL parser
# ════════════════════════════════════════════════════════════════════════════════

TOKEN_RE = re.compile(r'"[^"]*"|[{}();=,.\-]|[A-Za-z_][A-Za-z0-9_]*|\d+')


def norm_type(raw):
    t = re.sub(r'[^a-z]', '', str(raw or '').lower())
    if 'softgoal' in t or 'qualitygoal' in t or 'nfr' in t:
        return 'SoftGoal'
    if 'goal' in t:
        return 'Goal'
    if 'task' in t:
        return 'Task'
    if 'resource' in t:
        return 'Resource'
    if 'belief' in t:
        return 'Belief'
    return 'Goal'


def _strip_comments(text):
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    return text


class _Tok:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self, off=0):
        idx = self.i + off
        return self.toks[idx] if 0 <= idx < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def eat(self, v):
        if self.peek() == v:
            self.i += 1

    def at_end(self):
        return self.i >= len(self.toks)


def _unquote(t):
    if t and t.startswith('"') and t.endswith('"'):
        return t[1:-1]
    return t


def parse_xgrl(text):
    """Returns {'actors': [...], 'elements': [...], 'links': [...]}."""
    actors, elements, links = [], [], []
    elem_by_xgrl = {}
    actor_by_xgrl = {}
    pending_links = []
    ambiguous_bare_ids = set()

    clean = _strip_comments(text)
    toks = TOKEN_RE.findall(clean)
    c = _Tok(toks)

    def read_block():
        c.eat('{')
        attrs, children, raw_statements = {}, [], []
        while not c.at_end() and c.peek() != '}':
            t = c.next()
            if t is None or t == '}':
                break
            if t == ';':
                continue
            if c.peek() == '=':
                c.eat('=')
                attrs[t] = _unquote(c.next())
                c.eat(';')
                continue
            nxt1 = c.peek()
            nxt2 = c.peek(1)
            if nxt1 and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', nxt1) and nxt2 == '{':
                kw = t
                elem_id = c.next()
                children.append({'kw': kw, 'id': elem_id, 'body': read_block()})
                continue
            stmt = [t]
            d = 0
            while not c.at_end():
                tok = c.peek()
                if tok == '{':
                    d += 1; stmt.append(c.next())
                elif tok == '}' and d > 0:
                    d -= 1; stmt.append(c.next())
                elif tok == '}' and d == 0:
                    break
                elif tok == ';' and d == 0:
                    c.next(); break
                else:
                    stmt.append(c.next())
            if len(stmt) > 1:
                raw_statements.append(stmt)
        c.eat('}')
        return {'attrs': attrs, 'children': children, 'rawStatements': raw_statements}

    def parse_link_stmt(stmt, actor_xgrl_id):
        if len(stmt) < 2:
            return
        link_kw = stmt[1]
        from_key = (actor_xgrl_id + '.' + stmt[0]) if actor_xgrl_id else stmt[0]
        if link_kw == 'decomposedBy':
            targets = [t for t in stmt[2:] if t not in (',', '{', '}')]
            for to_id in targets:
                to_key = (actor_xgrl_id + '.' + to_id) if actor_xgrl_id else to_id
                pending_links.append({'type': 'Decomposition', 'fromKey': from_key, 'toKey': to_key})
        elif link_kw == 'contributesTo':
            to_id = stmt[2] if len(stmt) > 2 else None
            to_key = (actor_xgrl_id + '.' + to_id) if actor_xgrl_id else to_id
            # Map GRL contribution keywords to numeric values, also accept plain integers
            _CONTRIB_KEYWORDS = {
                'make': 2, 'help': 1, 'somepositive': 1, 'some': 1,
                'unknown': 0, 'neutral': 0,
                'somenegative': -1, 'hurt': -1,
                'break': -2, 'breaklink': -2,
            }
            value = None
            j = 3
            while j < len(stmt):
                tok_low = stmt[j].lower()
                if tok_low in _CONTRIB_KEYWORDS:
                    value = _CONTRIB_KEYWORDS[tok_low]; break
                if stmt[j] == '-' and j + 1 < len(stmt) and re.match(r'^\d+$', stmt[j + 1]):
                    value = -int(stmt[j + 1]); break
                if re.match(r'^\d+$', stmt[j]):
                    value = int(stmt[j]); break
                j += 1
            pending_links.append({'type': 'Contribution', 'fromKey': from_key, 'toKey': to_key, 'value': value})
        elif link_kw == 'dependsOn':
            rest = [t for t in stmt[2:] if t not in ('{', '}')]
            to_key = ''.join(rest)
            if actor_xgrl_id and '.' not in to_key:
                to_key = actor_xgrl_id + '.' + to_key
            pending_links.append({'type': 'Dependency', 'fromKey': from_key, 'toKey': to_key})

    top_level_stmts = []

    while not c.at_end():
        t = c.next()
        if t != 'grl':
            continue
        c.next()  # model name
        c.eat('{')
        while not c.at_end() and c.peek() != '}':
            t2 = c.next()
            if t2 is None or t2 == '}':
                break
            if t2 == ';':
                continue
            if t2 == 'actor':
                actor_xgrl_id = c.next()
                body = read_block()
                aid = 'xg_a_%d' % len(actors)
                actor = {'id': aid, 'name': body['attrs'].get('name', actor_xgrl_id)}
                actors.append(actor)
                actor_by_xgrl[actor_xgrl_id] = actor
                for child in body['children']:
                    kw, xgrl_id, cb = child['kw'], child['id'], child['body']
                    etype = norm_type(kw)
                    if etype in ('Goal', 'SoftGoal', 'Task', 'Resource', 'Belief'):
                        ename = cb['attrs'].get('name', xgrl_id)
                        our_id = 'xg_e_%d' % len(elements)
                        decomp_raw = cb['attrs'].get('decompositionType')
                        raw_imp = cb['attrs'].get('importance')
                        try:
                            imp = int(raw_imp) if raw_imp is not None else None
                        except (ValueError, TypeError):
                            imp = raw_imp  # keep as-is for non-numeric values (e.g. "high")
                        elem = {
                            'id': our_id, 'name': ename, 'type': etype, 'actorId': aid,
                            'xgrlId': xgrl_id, 'attrs': cb['attrs'],
                            'importance': imp,
                            'decompositionType': decomp_raw.lower() if decomp_raw else None,
                            '_decompType': (decomp_raw or 'and').lower(),
                        }
                        elements.append(elem)
                        key = actor_xgrl_id + '.' + xgrl_id
                        elem_by_xgrl[key] = elem
                        if xgrl_id in ambiguous_bare_ids:
                            pass
                        elif xgrl_id in elem_by_xgrl:
                            del elem_by_xgrl[xgrl_id]
                            ambiguous_bare_ids.add(xgrl_id)
                        else:
                            elem_by_xgrl[xgrl_id] = elem
                for stmt in body['rawStatements']:
                    parse_link_stmt(stmt, actor_xgrl_id)
                continue
            stmt = [t2]
            d = 0
            while not c.at_end():
                tok = c.peek()
                if tok == '{':
                    d += 1; stmt.append(c.next())
                elif tok == '}' and d > 0:
                    d -= 1; stmt.append(c.next())
                elif tok == '}' and d == 0:
                    break
                elif tok == ';' and d == 0:
                    c.next(); break
                else:
                    stmt.append(c.next())
            if len(stmt) > 2:
                top_level_stmts.append(stmt)
        c.eat('}')

    for stmt in top_level_stmts:
        condensed = []
        j = 0
        while j < len(stmt):
            if j + 1 < len(stmt) and stmt[j + 1] == '.':
                condensed.append(stmt[j] + '.' + (stmt[j + 2] if j + 2 < len(stmt) else ''))
                j += 3
            else:
                condensed.append(stmt[j])
                j += 1
        parse_link_stmt(condensed, None)

    def resolve_key(key):
        if key in elem_by_xgrl:
            return elem_by_xgrl[key]
        bare = key.split('.')[-1] if key and '.' in key else key
        return elem_by_xgrl.get(bare)

    for pl in pending_links:
        src = resolve_key(pl['fromKey'])
        tgt = resolve_key(pl['toKey']) if pl.get('toKey') else None
        if not src or not tgt:
            continue
        cross_actor = src['actorId'] != tgt['actorId']
        subtype = src.get('_decompType', 'and') if pl['type'] == 'Decomposition' else None
        links.append({
            'id': 'xg_l_%d' % len(links),
            'type': pl['type'], 'subtype': subtype,
            'value': pl.get('value'),
            'sourceId': src['id'], 'targetId': tgt['id'],
            'sourceActorId': src['actorId'], 'targetActorId': tgt['actorId'],
            'crossActor': cross_actor,
        })

    return {'actors': actors, 'elements': elements, 'links': links}


def describe(attrs):
    if not attrs:
        return ''
    return attrs.get('description') or attrs.get('desc') or attrs.get('definition') or attrs.get('note') or ''


def build_task_contexts(model):
    elements, links, actors = model['elements'], model['links'], model['actors']
    by_id = {e['id']: e for e in elements}
    actor_by_id = {a['id']: a for a in actors}
    tasks = [e for e in elements if e['type'] == 'Task']

    for task in tasks:
        actor = actor_by_id.get(task['actorId'])
        parent_goals = []
        for lnk in links:
            if lnk['type'] == 'Decomposition' and lnk['targetId'] == task['id']:
                g = by_id.get(lnk['sourceId'])
                if g:
                    parent_goals.append({'id': g['id'], 'name': g['name'], 'type': g['type']})
        resources = []
        for lnk in links:
            if lnk['sourceId'] == task['id'] or lnk['targetId'] == task['id']:
                outgoing = lnk['sourceId'] == task['id']
                other = by_id.get(lnk['targetId'] if outgoing else lnk['sourceId'])
                if other and other['type'] == 'Resource':
                    resources.append({
                        'id': other['id'], 'name': other['name'],
                        'relation': lnk['type'],
                        'direction': 'outgoing' if outgoing else 'incoming',
                    })
        task['context'] = {
            'name': task['name'],
            'description': describe(task['attrs']),
            'actor': {'id': actor['id'], 'name': actor['name']} if actor else None,
            'parentGoals': parent_goals,
            'resources': resources,
        }
        task.setdefault('readiness', {
            'status': 'pending',
            'A': None, 'S': None, 'R': None, 'D': None, 'C': None,
            'ars': None, 'level': None,
            'capability': None,
            'eligibility': None,
            'validation': None,
            'error': None,
        })

    return tasks


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1 — AI Readiness Assessment
# ════════════════════════════════════════════════════════════════════════════════

P1_CAPABILITIES = [
    'Classification', 'Prediction', 'Recommendation', 'Retrieval',
    'Summarization', 'Generation', 'Analytical Reasoning', 'None',
]

P1_SYSTEM = (
    'You are an expert requirements engineer specializing in AI-enabled systems. '
    'Your task is to assess the AI readiness of a GRL task by evaluating it along a '
    'set of structured dimensions. Use only the information provided and return the '
    'results in JSON format.'
)


def p1_build_prompt(task):
    ctx = task['context']
    parent_goals = '; '.join(f"{g['name']} ({g['type']})" for g in ctx['parentGoals']) or 'None specified'
    resources = '; '.join(f"{r['name']} ({r['relation']}, {r['direction']})" for r in ctx['resources']) or 'None specified'
    actor = ctx['actor']['name'] if ctx['actor'] else 'Unassigned'
    description = ctx['description'] or 'No description provided.'

    user = f"""Task Name: {ctx['name']}
Task Description: {description}
Parent Goals: {parent_goals}
Actor: {actor}
Resources: {resources}

Step 1 — For each of the following dimensions, reason through the task carefully, then provide a score from 1 to 5 and a brief justification:
1. Automation Potential (A): to what extent can the task outcome be produced without essential human judgment?
2. Input Structure (S): how structured, well-defined, and consistently formatted are the task inputs?
3. Repeatability (R): how frequently and consistently is this task performed across instances?
4. Reasoning Fit (D): how well do the task's reasoning demands align with the strengths of the most suitable AI technique (e.g., language, vision, or predictive models) rather than tacit domain judgment?

Step 2 — Evaluate the degree of external regulatory, ethical, legal, or organizational constraint associated with using AI for this specific task. Constraint Level (C): Provide a score from 1 to 5 and a brief justification.

Important:
Assess only constraints on AI adoption, not task complexity, importance, or automation potential.
Do not assign a high score simply because the task belongs to a regulated domain.
Focus on whether laws, liability concerns, governance policies, or mandatory human oversight restrict AI use for this task.

Scoring:
1: Minimal constraints; AI can be used freely.
2: Minor oversight or policy considerations.
3: Human review, validation, or approval is required.
4: Specific legal, regulatory, or organizational requirements significantly restrict AI use.
5: AI use is prohibited, highly restricted by regulation, or creates unacceptable legal or ethical risk.

Step 3 — Select the single AI capability category that best characterizes the task: Classification, Prediction, Recommendation, Retrieval, Summarization, Generation, or Analytical Reasoning. If the task does not correspond to any AI capability, return "None". Provide a confidence score from 0 to 100 and a brief justification.

Reason through each step internally before assigning scores, but do not include that reasoning in your output. Return ONLY the following JSON object — no prose, no markdown code fences, no additional keys, no commentary before or after it:

{{
  "A": {{"score": 1, "justification": "..."}},
  "S": {{"score": 1, "justification": "..."}},
  "R": {{"score": 1, "justification": "..."}},
  "D": {{"score": 1, "justification": "..."}},
  "C": {{"score": 1, "justification": "..."}},
  "capability": {{"category": "...", "confidence": 0, "justification": "..."}}
}}"""
    return P1_SYSTEM, user


def normalize_capability(cat):
    if not isinstance(cat, str):
        return None
    norm = cat.strip().lower()
    for c in P1_CAPABILITIES:
        if c.lower() == norm:
            return c
    return None


def p1_validate_json(obj):
    if not isinstance(obj, dict):
        raise ValueError('Response was not a JSON object.')
    for k in ('A', 'S', 'R', 'D', 'C'):
        d = obj.get(k)
        if not isinstance(d, dict) or not isinstance(d.get('score'), (int, float)):
            raise ValueError(f'Missing or invalid "{k}" dimension in response.')
        d['score'] = min(5, max(1, round(d['score'])))
        d['justification'] = d.get('justification') if isinstance(d.get('justification'), str) else ''
    cap = obj.get('capability')
    if not isinstance(cap, dict):
        raise ValueError('Missing "capability" object in response.')
    cat = normalize_capability(cap.get('category'))
    if not cat:
        raise ValueError(f'Unrecognized capability category "{cap.get("category")}".')
    cap['category'] = cat
    conf = cap.get('confidence')
    cap['confidence'] = min(100, max(0, round(conf))) if isinstance(conf, (int, float)) else 0
    cap['justification'] = cap.get('justification') if isinstance(cap.get('justification'), str) else ''
    return obj


def p1_compute_eligibility(r):
    ars = (r['A']['score'] + r['S']['score'] + r['R']['score'] + r['D']['score']) / 4.0
    if ars >= 4.0:
        level = 'High'
    elif ars >= 2.5:
        level = 'Medium'
    else:
        level = 'Low'
    if level == 'Low':
        eligibility = 'retained_low'
    elif r['C']['score'] >= 4:
        eligibility = 'excluded'
    elif r['C']['score'] == 3:
        eligibility = 'partially_constrained'
    else:
        eligibility = 'carry_forward'
    return ars, level, eligibility


def p1_assess_task(task, api_key):
    """Runs Phase 1 assessment for one task. Returns True on success."""
    r = task['readiness']
    r['status'] = 'running'
    r['error'] = None
    system, user = p1_build_prompt(task)

    def try_once(u):
        raw = llm_call(system, u, api_key, P1_MAX_TOKENS)
        parsed = p1_validate_json(extract_json(raw))
        r['A'], r['S'], r['R'], r['D'], r['C'] = (
            parsed['A'], parsed['S'], parsed['R'], parsed['D'], parsed['C'])
        r['capability'] = parsed['capability']
        r['status'] = 'done'
        r['error'] = None
        ars, level, eligibility = p1_compute_eligibility(r)
        r['ars'], r['level'], r['eligibility'] = ars, level, eligibility
        if eligibility == 'partially_constrained':
            r['validation'] = {'required': True, 'validated': False, 'validatedAt': None, 'note': ''}
        else:
            r['validation'] = None

    try:
        try_once(user)
        return True
    except LLMError as err:
        reason = f'request failed ({err})'
    except Exception as err:
        reason = str(err)

    retry_user = (user + f'\n\nYour previous response could not be used ({reason}). '
                         'Return ONLY the JSON object specified above — no prose, '
                         'no markdown fences, no commentary, no explanation.')
    try:
        try_once(retry_user)
        return True
    except Exception as err2:
        r['status'] = 'error'
        r['error'] = str(err2)
        return False


def p1_build_report(file_name, tasks):
    return {
        'fileName': file_name,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'tasks': [
            {
                'id': t['id'],
                'name': t['context']['name'],
                'description': t['context']['description'],
                'actor': t['context']['actor'],
                'parentGoals': t['context']['parentGoals'],
                'resources': t['context']['resources'],
                'readiness': t['readiness'],
            }
            for t in tasks
        ],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2 — AI Transformation Patterns
# ════════════════════════════════════════════════════════════════════════════════

PATTERNS = {
    'P1': {
        'capabilities': ['Analytical Reasoning'],
        'traditionalActivity': 'Analyze documents, requests, or cases',
        'aiTransformation': 'AI-Assisted Analysis',
        'governance': ['Explainability', 'Accountability', 'Robustness'],
    },
    'P2': {
        'capabilities': ['Prediction', 'Recommendation'],
        'traditionalActivity': 'Apply predefined business rules',
        'aiTransformation': 'AI Prediction / Recommendation',
        'governance': ['Fairness', 'Explainability', 'Robustness'],
    },
    'P3': {
        'capabilities': ['Retrieval'],
        'traditionalActivity': 'Search repositories or knowledge bases',
        'aiTransformation': 'AI-Assisted Retrieval',
        'governance': ['Privacy', 'Explainability', 'Robustness'],
    },
    'P4': {
        'capabilities': ['Classification'],
        'traditionalActivity': 'Categorize requests, documents, or events',
        'aiTransformation': 'AI Classification',
        'governance': ['Fairness', 'Explainability'],
    },
    'P5': {
        'capabilities': ['Generation', 'Summarization'],
        'traditionalActivity': 'Generate reports, summaries, or documents',
        'aiTransformation': 'Generative AI Assistance',
        'governance': ['Accountability', 'Explainability', 'Robustness'],
    },
}

CAPABILITY_TO_PATTERN = {}
for _pid, _info in PATTERNS.items():
    for _cap in _info['capabilities']:
        CAPABILITY_TO_PATTERN[_cap] = _pid

GOVERNANCE_REQUIREMENTS = ['Explainability', 'Fairness', 'Accountability', 'Privacy', 'Robustness']


def p2_pattern_for(task):
    cat = (task.get('readiness') or {}).get('capability', {}).get('category')
    return CAPABILITY_TO_PATTERN.get(cat)


def p2_eligible_tasks(report):
    out = []
    for t in report['tasks']:
        r = t.get('readiness') or {}
        if r.get('status') != 'done':
            continue
        if r.get('eligibility') not in ('carry_forward', 'partially_constrained'):
            continue
        cap = r.get('capability') or {}
        if cap.get('category') and cap['category'] != 'None':
            out.append(t)
    return out


def p2_build_prompt(task):
    pid = p2_pattern_for(task)
    pattern = PATTERNS[pid]
    parent_goals = '; '.join(f"{g['name']} ({g['type']})" for g in task.get('parentGoals', [])) or 'None specified'
    actor = task['actor']['name'] if task.get('actor') else 'Unassigned'
    description = task.get('description') or 'No description provided.'
    resources_list = '\n'.join(
        f"- [{r['id']}] {r['name']} ({r['relation']}, {r['direction']})"
        for r in task.get('resources', [])
    ) or 'None specified'

    system = (
        'You are an expert requirements engineer specializing in AI-enabled systems.\n\n'
        'Your task is to generate an AI transformation specification for a GRL task, '
        'using the transformation pattern already selected for this task.\n\n'
        'Do not create new actors.\n\n'
        'Use only the governance requirements provided below. '
        'Do not select governance requirements outside this list.\n\n'
        f'Available Governance Requirements (for this task\'s pattern only): {", ".join(pattern["governance"])}.\n\n'
        'Return ONLY a JSON object — no prose, no markdown code fences, '
        'no additional keys, no commentary before or after it.'
    )

    user = f"""Task Name: {task['name']}
Task Description: {description}
Parent Goals: {parent_goals}
Actor: {actor}
Resources (each prefixed with its id in brackets):
{resources_list}

Capability Category: {task['readiness']['capability']['category']}
Transformation Pattern: {pid} — {pattern['aiTransformation']} (traditional activity being replaced: "{pattern['traditionalActivity']}")

Generate:
1. One AI-enabled task: a name and a description of the task in its AI-enabled context.
2. The new supporting resource(s) required by the AI-enabled task, when they are strongly required. Empty list if none are needed.
3. For each original resource listed above, by its id, indicate whether it should be retained or replaced, with a brief rationale.
4. Applicable governance requirements for this transformed task, selected ONLY from the list above, each with a brief justification.

Return ONLY the following JSON object:

{{
  "aiTask": {{"name": "...", "description": "..."}},
  "resources": [{{"name": "...", "description": "..."}}],
  "resourceMapping": [{{"resourceId": "...", "decision": "retain or replace", "rationale": "..."}}],
  "governance": [{{"requirement": "...", "justification": "..."}}]
}}"""
    return system, user


def p2_normalize_governance(req, allowed_list):
    if not isinstance(req, str):
        return None
    norm = req.strip().lower()
    for g in (allowed_list or GOVERNANCE_REQUIREMENTS):
        if g.lower() == norm:
            return g
    return None


def p2_validate_json(obj, task):
    if not isinstance(obj, dict):
        raise ValueError('Response was not a JSON object.')
    ai_task = obj.get('aiTask')
    if (not isinstance(ai_task, dict)
            or not isinstance(ai_task.get('name'), str) or not ai_task.get('name', '').strip()
            or not isinstance(ai_task.get('description'), str) or not ai_task.get('description', '').strip()):
        raise ValueError('Missing or invalid "aiTask" (name/description) in response.')

    resources = []
    for res in obj.get('resources') or []:
        if isinstance(res, dict) and isinstance(res.get('name'), str) and res['name'].strip():
            resources.append({
                'name': res['name'].strip(),
                'description': res.get('description') if isinstance(res.get('description'), str) else '',
            })

    id_by_context = {r['id']: r['name'] for r in task.get('resources', [])}
    resource_mapping = []
    for m in obj.get('resourceMapping') or []:
        if isinstance(m, dict) and m.get('resourceId') in id_by_context:
            decision = ('replace'
                        if isinstance(m.get('decision'), str)
                        and m['decision'].strip().lower().startswith('replace')
                        else 'retain')
            resource_mapping.append({
                'resourceId': m['resourceId'],
                'originalName': id_by_context[m['resourceId']],
                'decision': decision,
                'rationale': m.get('rationale') if isinstance(m.get('rationale'), str) else '',
            })

    pid = p2_pattern_for(task)
    pattern = PATTERNS.get(pid)
    allowed_governance = pattern['governance'] if pattern else GOVERNANCE_REQUIREMENTS
    governance = []
    for g in obj.get('governance') or []:
        if not isinstance(g, dict):
            continue
        req = p2_normalize_governance(g.get('requirement'), allowed_governance)
        if req:
            governance.append({
                'requirement': req,
                'justification': g.get('justification') if isinstance(g.get('justification'), str) else '',
            })

    return {
        'aiTask': {'name': ai_task['name'].strip(), 'description': ai_task['description'].strip()},
        'resources': resources,
        'resourceMapping': resource_mapping,
        'governance': governance,
    }


def p2_transform_task(task, api_key):
    """Runs Phase 2 transformation spec generation for one task. Returns True on success."""
    t = task.setdefault('transformation', {
        'status': 'pending', 'pattern': None, 'aiTask': None,
        'resources': [], 'resourceMapping': [], 'governance': [],
        'requiresValidation': False, 'error': None,
    })
    t['status'] = 'running'
    t['error'] = None
    system, user = p2_build_prompt(task)

    def try_once(u):
        raw = llm_call(system, u, api_key, P2_MAX_TOKENS)
        parsed = p2_validate_json(extract_json(raw), task)
        t['pattern'] = p2_pattern_for(task)
        t['aiTask'] = parsed['aiTask']
        t['resources'] = parsed['resources']
        t['resourceMapping'] = parsed['resourceMapping']
        t['governance'] = parsed['governance']
        t['requiresValidation'] = task['readiness']['eligibility'] == 'partially_constrained'
        t['status'] = 'done'
        t['error'] = None

    try:
        try_once(user)
        return True
    except LLMError as err:
        reason = f'request failed ({err})'
    except Exception as err:
        reason = str(err)

    retry_user = (user + f'\n\nYour previous response could not be used ({reason}). '
                         'Return ONLY the JSON object specified above — no prose, '
                         'no markdown fences, no commentary, no explanation.')
    try:
        try_once(retry_user)
        return True
    except Exception as err2:
        t['status'] = 'error'
        t['error'] = str(err2)
        return False


def p2_build_export(file_name, tasks):
    done = [t for t in tasks if t.get('transformation', {}).get('status') == 'done']
    return {
        'fileName': file_name,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'transformations': [
            {
                'taskId': t['id'],
                'taskName': t['name'],
                'pattern': t['transformation']['pattern'],
                'requiresValidation': t['transformation']['requiresValidation'],
                'aiTask': t['transformation']['aiTask'],
                'resources': t['transformation']['resources'],
                'resourceMapping': t['transformation']['resourceMapping'],
                'governance': t['transformation']['governance'],
            }
            for t in done
        ],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Phase 3 — LLM-Based TGRL Transformation
# ════════════════════════════════════════════════════════════════════════════════

P3_SYSTEM = (
    'You are an expert requirements engineer specializing in GRL and TGRL.\n\n'
    'Your task is to transform a structured TGRL specification into a new AI-enabled TGRL specification.\n\n'
    'Use only the provided original TGRL specification and the transformation specifications.\n\n'
    'Do not introduce new actors.\n\n'
    'Keep the output in valid TGRL syntax.'
)


def p3_build_prompt(original_tgrl, specs):
    user = f"""Original TGRL Specification:
{original_tgrl}

Transformation Specifications:
{json.dumps(specs, indent=2)}

Instructions
1. Replace each original task by its AI-enabled counterpart.
2. Update or replace the related resources.
3. Add a dependsOn link between each AI-enabled task and each of its supporting resources (newly added or retained).
4. Add the governance softgoals associated with each transformed task. Each governance softgoal's name has to be one of the following: 'Explainability', 'Fairness', 'Accountability', 'Privacy', 'Robustness'. Each softgoal should be unique within the actor.
5. Add a contributesTo link between each AI-enabled task and each of its governance softgoals; assign each link a positive or negative contribution value depending on whether the task supports or threatens that governance requirement, and provide a brief justification.
6. Preserve all unchanged actors, goals, softgoals, decompositions, and dependencies.
7. Return the full transformed TGRL specification only."""
    return user


def p3_check_plausibility(text):
    """Returns None if the output looks structurally valid, or a warning string."""
    clean = _strip_comments(text)
    toks = TOKEN_RE.findall(clean)
    if 'grl' not in toks:
        return 'No top-level "grl" model declaration found.'
    if toks.count('{') != toks.count('}'):
        return f'Unbalanced braces ({toks.count("{")} "{{" vs {toks.count("}")} "}}").'
    if 'actor' not in toks:
        return 'No "actor" block found.'
    return None


def p3_transform(original_tgrl, specs, api_key):
    """Single whole-model LLM call. Returns (output_text, warning_or_None)."""
    user = p3_build_prompt(original_tgrl, specs)
    try:
        raw = llm_call(P3_SYSTEM, user, api_key, P3_MAX_TOKENS)
    except LLMError as err:
        retry_user = (user + f'\n\nYour previous response could not be used '
                             f'(request failed: {err}). Return ONLY the full '
                             'transformed TGRL specification — no prose, no '
                             'commentary, no markdown fences.')
        raw = llm_call(P3_SYSTEM, retry_user, api_key, P3_MAX_TOKENS)

    output = strip_fences(raw)
    warning = p3_check_plausibility(output)
    return output, warning


# ════════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GRL AI Transformer — runs all three transformation phases on a .xgrl model.'
    )
    parser.add_argument('input', help='Path to the .xgrl / TGRL model file')
    args = parser.parse_args()

    # --- API key ---
    # Prefer system/user environment variable; fall back to .env file if available.
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('OPENAI_API_KEY', '').strip()
        except ImportError:
            pass  # python-dotenv not installed — that's fine, system env is the primary source
    if not api_key:
        print('Error: OPENAI_API_KEY environment variable is not set.', file=sys.stderr)
        sys.exit(1)

    # --- Setup ---
    input_path = args.input
    with open(input_path, 'r', encoding='utf-8') as f:
        original_tgrl = f.read()

    file_name = os.path.basename(input_path)
    prefix = safe_base_name(file_name)
    output_dir = os.path.dirname(os.path.abspath(input_path))
    print(f'Model: {file_name}')
    print(f'LLM:   {MODEL}\n')

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print('━━━ Phase 1: AI Readiness Assessment ━━━')
    model = parse_xgrl(original_tgrl)
    tasks = build_task_contexts(model)
    print(f'Found {len(tasks)} task(s).\n')

    for idx, task in enumerate(tasks, start=1):
        print(f'  [{idx}/{len(tasks)}] Assessing "{task["name"]}" ...', end=' ', flush=True)
        ok = p1_assess_task(task, api_key)
        if ok:
            r = task['readiness']
            print(f'ARS={r["ars"]:.2f}  level={r["level"]}  eligibility={r["eligibility"]}  capability={r["capability"]["category"]}')
        else:
            print(f'ERROR: {task["readiness"]["error"]}')

    p1_report = p1_build_report(file_name, tasks)
    p1_path = os.path.join(output_dir, f'{prefix}_phase1_readiness.json')
    with open(p1_path, 'w', encoding='utf-8') as f:
        json.dump(p1_report, f, indent=2)
    print(f'\nPhase 1 complete → {p1_path}\n')

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print('━━━ Phase 2: AI Transformation Patterns ━━━')
    eligible = p2_eligible_tasks(p1_report)
    print(f'{len(eligible)} task(s) eligible for transformation.\n')

    if not eligible:
        print('No eligible tasks — skipping Phase 2 and Phase 3.')
        sys.exit(0)

    for idx, task in enumerate(eligible, start=1):
        pid = p2_pattern_for(task)
        label = PATTERNS[pid]['aiTransformation']
        print(f'  [{idx}/{len(eligible)}] "{task["name"]}" → pattern {pid} ({label}) ...', end=' ', flush=True)
        ok = p2_transform_task(task, api_key)
        if ok:
            print(f'→ "{task["transformation"]["aiTask"]["name"]}"')
        else:
            print(f'ERROR: {task["transformation"]["error"]}')

    p2_export = p2_build_export(file_name, eligible)
    p2_path = os.path.join(output_dir, f'{prefix}_phase2_transformations.json')
    with open(p2_path, 'w', encoding='utf-8') as f:
        json.dump(p2_export, f, indent=2)
    done_count = sum(1 for t in eligible if t.get('transformation', {}).get('status') == 'done')
    print(f'\nPhase 2 complete ({done_count}/{len(eligible)} specs generated) → {p2_path}\n')

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print('━━━ Phase 3: LLM-Based TGRL Transformation ━━━')
    specs = p2_export.get('transformations', [])
    if not specs:
        print('No completed transformation specs — cannot run Phase 3.', file=sys.stderr)
        sys.exit(1)

    print(f'Transforming model using {len(specs)} spec(s) ...')
    try:
        output, warning = p3_transform(original_tgrl, specs, api_key)
    except LLMError as err:
        print(f'Error: Phase 3 transformation failed: {err}', file=sys.stderr)
        sys.exit(1)

    p3_path = os.path.join(output_dir, f'{prefix}_phase3_transformed.xgrl')
    with open(p3_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print('Phase 3 complete.')
    if warning:
        print(f'  Note: structural plausibility check: {warning}')
        print('  (review the output file — the transformation may still be correct)')
    print(f'→ {p3_path}')

    print('\n━━━ All done ━━━')
    print(f'  {p1_path}')
    print(f'  {p2_path}')
    print(f'  {p3_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
