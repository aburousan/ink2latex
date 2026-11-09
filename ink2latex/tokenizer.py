import json
import re
from collections import Counter
from pathlib import Path

SPECIAL = ["<pad>", "<s>", "</s>"]
TOKEN = re.compile(r"\\[a-zA-Z]+|\\[^\s]|[^\s]")


# Commands whose arguments are always braced in MathWriting's normalised labels.
ONE_ARG = {"\\sqrt", "\\vec", "\\hat", "\\bar", "\\tilde", "\\dot", "\\ddot", "\\overline", "\\underline",
           "\\widehat", "\\widetilde", "\\overrightarrow", "\\mathbf", "\\mathrm", "\\mathcal", "\\mathbb",
           "\\boldsymbol", "\\text", "\\operatorname"}
TWO_ARGS = {"\\frac", "\\dfrac", "\\tfrac", "\\binom"}


def _match(tokens, i, open_="{", close="}"):
    """Index of the bracket closing tokens[i] (len(tokens) if it never closes)."""
    depth = 0
    for j in range(i, len(tokens)):
        depth += tokens[j] == open_
        depth -= tokens[j] == close
        if depth == 0:
            return j
    return len(tokens)


def _argument(tokens, i):
    """One argument starting at i, braced: {..} kept, a lone token or command (with its own arguments) wrapped."""
    if i >= len(tokens):
        return [], i
    if tokens[i] == "{":
        j = _match(tokens, i)
        return ["{", *canonical(tokens[i + 1:j]), "}"], j + 1
    if tokens[i] in ONE_ARG or tokens[i] in TWO_ARGS:
        inner, k = _command(tokens, i)
        return ["{", *inner, "}"], k
    return ["{", tokens[i], "}"], i + 1


def _command(tokens, i):
    t, out, k = tokens[i], [tokens[i]], i + 1
    if t == "\\sqrt" and k < len(tokens) and tokens[k] == "[":
        j = _match(tokens, k, "[", "]")
        out += ["[", *canonical(tokens[k + 1:j]), "]"]
        k = j + 1
    for _ in range(2 if t in TWO_ARGS else 1):
        arg, k = _argument(tokens, k)
        out += arg
    return out, k


def canonical(tokens):
    """MathWriting's style for any label: every script, fraction and accent argument in braces
    (older labels wrote x^2 and \\frac a b). Already-braced labels come back unchanged."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("^", "_"):
            arg, i = _argument(tokens, i + 1)
            out += [t, *arg]
        elif t in ONE_ARG or t in TWO_ARGS:
            part, i = _command(tokens, i)
            out += part
        elif t == "{":
            j = _match(tokens, i)
            out += ["{", *canonical(tokens[i + 1:j]), *(["}"] if j < len(tokens) else [])]
            i = j + 1
        else:
            out.append(t)
            i += 1
    return out


def tokenize(latex):
    return canonical(TOKEN.findall(latex) if isinstance(latex, str) else list(latex))


FUNCTION_NAMES = ("arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "lim", "sin", "cos", "tan", "cot", "sec",
                  "csc", "log", "exp", "max", "min", "det", "sup", "inf", "ln")


def tidy(tokens):
    """Function names spelled in single letters (l i m, s i n) become their commands (\\lim, \\sin) unless a letter
    comes right before (a s i n stays): photos and some hands give the letters. A letter after is the argument
    (s i n x is \\sin x); longer names are tried first (sinh before sin)."""
    out, i = [], 0
    letter = lambda k: 0 <= k < len(tokens) and len(tokens[k]) == 1 and tokens[k].isalpha()
    while i < len(tokens):
        for name in FUNCTION_NAMES:
            n = len(name)
            if tokens[i:i + n] == list(name) and not letter(i - 1):
                out.append("\\" + name); i += n
                break
        else:
            out.append(tokens[i]); i += 1
    return out


def to_latex(tokens):
    """Compact LaTeX to paste or typeset: a space only where a command name meets a letter (\\alpha x)."""
    out = ""
    for token in tokens:
        if re.search(r"\\[A-Za-z]+$", out) and token[:1].isalpha():
            out += " "
        out += token
    return out


def detokenize(tokens):
    # A separator prevents e.g. ['\\alpha', 'x'] from becoming the command \\alphax.
    return " ".join(tokens)


class Tokenizer:
    pad, bos, eos = 0, 1, 2

    def __init__(self, tokens):
        self.tokens = list(tokens)
        if self.tokens[:3] != SPECIAL or len(set(tokens)) != len(tokens):
            raise ValueError("Invalid vocabulary")
        if len(tokens) > 32767:
            raise ValueError("Vocabulary exceeds int16 storage")
        self.ids = {token: i for i, token in enumerate(tokens)}

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def build(cls, labels, min_count=20):
        counts = Counter(t for label in labels for t in tokenize(label))
        return cls(SPECIAL + sorted(t for t, n in counts.items() if n >= min_count))

    def encode(self, label, *, allow_unknown=False):
        tokens = tokenize(label)
        if allow_unknown:
            return [self.ids.get(t, -1) for t in tokens]
        return [self.ids[t] for t in tokens]

    def decode(self, ids):
        result = []
        for i in ids:
            if i == self.eos:
                break
            if i >= 3:
                result.append(self.tokens[i])
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["tokens"])
