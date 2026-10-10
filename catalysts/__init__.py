"""
catalysts — news-first event intelligence for Prediction Engine v2 (shadow).

Chain: provider article -> sanitise -> deduplicate / cluster into a canonical
event -> classify (category, sentiment, materiality, novelty, expectedness)
-> factual entities -> inferred exposures (versioned transmission
hypotheses) -> point-in-time assessment per stock -> NEUTRAL / NO_CALL unless
the evidence is strong, fresh, credible, uncontradicted and not yet priced.

News text is untrusted data: it is sanitised, length-limited and only ever
matched against fixed vocabularies. It is never executed, evaluated or used
to build instructions, queries or prompts.
"""
CLASSIFIER_VERSION = "news-rules-0.1"
NEWS_RULE_VERSION = "news-v0.1"
