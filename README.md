# Belief ROC Studio — live demo

**Try it here: https://khurramyamin.github.io/belief-roc-studio-demo/**

A point-and-click tool for tuning an AI's clinical decisions to a hospital's
own priorities: load a holdout set of past cases, elicit the AI's
*probabilities* (never its decisions), build the ROC curve, state how you weigh
missed cases against unnecessary actions, and get the best decision cut-off —
then apply it automatically to new cases.

This page is a **demo**: everything runs in your browser with a *simulated* AI
(practice mode) and a built-in synthetic example dataset — no data leaves the
page. The full (also free) downloadable tool additionally connects to ChatGPT
(OpenAI), Claude (Anthropic), or Gemini (Google) with your own API key.

Method: Yamin et al., *"Talk is not cheap: Eliciting beliefs and utilities is
necessary for trustworthy medical AI decisions."*
