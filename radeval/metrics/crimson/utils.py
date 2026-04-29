"""JSON parsing utilities for CRIMSON scoring.

Ported from the standalone CRIMSON package to provide robust JSON parsing
for LLM responses that may contain malformed JSON.
"""

import json
import re

# ---------------------------------------------------------------------------
# Regex for fixing invalid JSON backslash escapes
# ---------------------------------------------------------------------------

# Fix invalid JSON backslash escapes (e.g. \_ or \L) that the model may
# produce when echoing back input text.  Valid JSON escapes
# (" \\ \/ \b \f \n \r \t \uXXXX) are left untouched.
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\\//bfnrtu])')

# ---------------------------------------------------------------------------
# Regex for removing orphan keys in JSON objects
# ---------------------------------------------------------------------------

# Matches a bare string value inside a JSON object — a quoted string preceded
# by a comma and followed by a comma or closing brace, but NOT followed by a
# colon.  Example:  ,"pred_old",  or  ,"pred_old"}
# The leading comma is consumed so removal doesn't leave double commas.
_ORPHAN_KEY_AFTER_COMMA_RE = re.compile(r',\s*"[^"]*"\s*(?=\s*[,}])(?!\s*:)')
# Orphan as the first item in an object:  {"orphan","key":"val"}
_ORPHAN_KEY_AT_START_RE = re.compile(r'(?<=\{)\s*"[^"]*"\s*,\s*(?=")')

# Malformed array-object hybrid:  ["":["R1"]  ->  ["R1"]
_MALFORMED_ARRAY_OBJ_RE = re.compile(r'\[""\s*:\s*\[')

# Missing opening quote on a JSON key:  ,attribute_errors":[  ->  ,"attribute_errors":[
_MISSING_OPEN_QUOTE_RE = re.compile(
    r'([\]}"0-9]\s*,\s*|{\s*)([a-zA-Z_][a-zA-Z0-9_]*")'
)


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _dedupe_keys(pairs):
    """JSON object_pairs_hook that keeps the first occurrence of duplicate keys."""
    result = {}
    for key, value in pairs:
        if key not in result:
            result[key] = value
    return result


def _loads(text):
    """json.loads with duplicate-key handling (keeps first occurrence)."""
    return json.loads(text, object_pairs_hook=_dedupe_keys)


def _is_structural_quote(text, i):
    """Check if the quote at position *i* is likely a JSON structural delimiter.

    Returns True for quotes that delimit JSON keys/values (after ``:``, before
    ``:`` etc.) and False for quotes that are part of prose inside a string
    value (e.g. the model quoting a phrase in an explanation).
    """
    before = text[i - 1] if i > 0 else ''
    after = text[i + 1] if i + 1 < len(text) else ''
    # Quotes right after : [ { are always structural (opening a value/key)
    if before in (':', '[', '{'):
        return True
    # Quotes right before : ] } are always structural (closing a key/value)
    if after in (':', ']', '}'):
        return True
    # For comma-adjacent quotes, check deeper context.
    if before == ',':
        pre_comma = text[i - 2] if i >= 2 else ''
        return pre_comma in ('"', ']', '}') or pre_comma.isdigit()
    if after == ',':
        post_comma = text[i + 2] if i + 2 < len(text) else ''
        return post_comma in ('"', '[', '{') or post_comma.isdigit()
    return False


def _fix_unescaped_quotes(text, max_attempts=50):
    """Iteratively escape unescaped double-quotes that break JSON parsing.

    When the model uses literal " inside a JSON string value (e.g. to quote
    a phrase in an explanation), the JSON parser sees it as a string
    terminator.  This function locates each offending quote by parsing,
    catching the error position, searching backwards for a non-structural
    quote, escaping it, and retrying.
    """
    for _ in range(max_attempts):
        try:
            return _loads(text)
        except json.JSONDecodeError as e:
            pos = e.pos
            search_start = max(0, pos - 5)
            found = False
            for i in range(pos, search_start - 1, -1):
                if i < len(text) and text[i] == '"' and (i == 0 or text[i - 1] != '\\'):
                    if _is_structural_quote(text, i):
                        continue
                    text = text[:i] + '\\"' + text[i + 1:]
                    found = True
                    break
            if not found:
                raise
    raise json.JSONDecodeError("Max quote-fix attempts reached", text, 0)


def _fix_orphan_keys(text):
    """Remove bare string values inside JSON objects (orphan keys with no value).

    The model sometimes hallucinates partial key fragments (e.g. ``"pred_old"``)
    that appear as bare values in objects, producing invalid JSON like::

        {"ref_id":"R3","pred_old","pred_id":"P2"}

    This function strips them so the JSON becomes parsable.
    """
    text = _ORPHAN_KEY_AFTER_COMMA_RE.sub('', text)
    text = _ORPHAN_KEY_AT_START_RE.sub('', text)
    return text


def parse_json_response(response, batch_idx=None):
    """Parse model response as JSON, applying progressive fixes.

    Fix pipeline (each step only attempted if prior steps fail):
    1. Raw parse
    2. Remove orphan keys (bare strings in object context)
    3. Fix invalid backslash escapes (``\\L``, ``\\_``, etc.)
    4. Fix unescaped double-quotes inside string values

    Args:
        response: Raw model output string.
        batch_idx: Optional index for error context in batch evaluation.

    Returns:
        Parsed JSON object (dict).

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    # 0. Pre-parse text fixes
    # Escape curly/smart quotes — they always appear inside JSON string values
    response = response.replace('\u201c', '\\"').replace('\u201d', '\\"')
    response = response.replace('\u2018', "\\'").replace('\u2019', "\\'")
    # Fix malformed array-object hybrids:  ["":["R1"] -> ["R1"]
    response = _MALFORMED_ARRAY_OBJ_RE.sub('[', response)
    # Fix missing opening quote on keys:  ,attribute_errors"  ->  ,"attribute_errors"
    response = _MISSING_OPEN_QUOTE_RE.sub(r'\1"\2', response)
    # 1. Raw
    try:
        return _loads(response)
    except json.JSONDecodeError:
        pass
    # 2. Orphan keys (bare strings in object context, e.g. "pred_old")
    deorphaned = _fix_orphan_keys(response)
    try:
        return _loads(deorphaned)
    except json.JSONDecodeError:
        pass
    # 3. Invalid backslash escapes
    escaped = _INVALID_ESCAPE_RE.sub(r'\\\\', response)
    try:
        return _loads(escaped)
    except json.JSONDecodeError:
        pass
    # 4. Unescaped quotes (try on both raw and escape-fixed variants)
    for text in (response, escaped):
        try:
            return _fix_unescaped_quotes(text)
        except (json.JSONDecodeError, ValueError):
            pass
    ctx = f" for batch item {batch_idx}" if batch_idx is not None else ""
    raise ValueError(
        f"Failed to parse model response as JSON{ctx}\n"
        f"Response ({len(response)} chars):\n{response}"
    )
