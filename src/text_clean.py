"""Text cleaning helpers."""

import re

# Almost all real articles are Reuters wire stories that begin with a dateline
# like "WASHINGTON (Reuters) - ...". That tag basically gives away the label, so
# we strip the dateline and any other mention of the agency name.
_DATELINE_RE = re.compile(r"^\s*[A-Za-z .,'/\-]{0,60}?\(reuters\)\s*[-–—]\s*", re.IGNORECASE)
_REUTERS_RE = re.compile(r"\(?\breuters\b\)?", re.IGNORECASE)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z']+")
_MULTISPACE_RE = re.compile(r"\s+")


def strip_source_artifacts(text):
    text = _DATELINE_RE.sub(" ", text)
    text = _REUTERS_RE.sub(" ", text)
    return text


def clean_text(text, strip_artifacts=True):
    """Lower-case, strip source tags/urls, keep letters only, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if strip_artifacts:
        text = strip_source_artifacts(text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text
