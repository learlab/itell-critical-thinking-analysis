# 10-Minute Slide Deck Outline — ITELL Critical Thinking Analysis

Timing: 12 slides × ~50s each, ~1 min slack for transitions.

## Slide 1 — Title (30s)
- *Does Dialogic AI Feedback Drive Deeper Summary Revision?*
- Subtitle: Evaluating STAIRS vs. random-reread control in an economics textbook
- Author, affiliation, date

## Slide 2 — Motivation & Research Question (1 min)
- iTELL embeds summary-writing checkpoints in textbook reading
- Question: When a student's summary falls short, does *interactive AI feedback* (STAIRS) produce more substantive revisions than simply asking them to reread?
- Why it matters: scalable formative feedback is a core promise of LLM tutoring

## Slide 3 — STAIRS vs. Control (1 min)
- STAIRS: dialogic chat explaining gaps, pointing to relevant passages
- Control: random-reread prompt (no targeted feedback)
- Both conditions trigger on the same rubric-based failure

## Slide 4 — Study Design & Data (1 min)
- Prolific study, N = 372 students, economics textbook (7 chapters)
- 267 revision pairs (first submission → earliest resubmission, same page)
  - 157 STAIRS, 110 control
- Within-subject paired design

## Slide 5 — Features Computed (1 min)
- **Structure:** word count, sentence count, MTLD (lexical diversity)
- **iTELL rubric:** content, language, relevance, containment
- **Semantic categories:** critical-thinking verbs (Bloom's-seeded, 11 terms), confusion (Empath)
  - spaCy `en_core_web_lg` centroid cosine similarity, normalized per 100 words
- **Edit distance:** revising vs. reworking

## Slide 6 — Analysis Approach (30s)
- Paired Cohen's d_z within each condition
- Mixed-effects regression with time × condition interaction
- Wilcoxon signed-rank as nonparametric check

## Slide 7 — Headline Results (1.5 min) ⭐
- **Figure:** `figs/headline_effect_sizes.png`
- Large STAIRS effects, small control effects on structural/content measures:
  - Word count: d_z = 0.93 vs 0.34 (+23.9 vs +5.6 words), interaction p < .001
  - Sentence count: d_z = 0.83 vs 0.33, interaction p < .001
  - iTELL content score: d_z = 0.94 vs 0.25, interaction p < .001
- MTLD ~flat in both → students add content without diluting quality

## Slide 8 — Full Feature Landscape (1 min)
- **Figure:** `figs/effect_size_heatmap.png`
- All 18 features, both conditions
- Shows *selectivity*: STAIRS moves the structural/content columns; abstract-reasoning columns stay quiet

## Slide 9 — The Null Results: Critical Thinking & Confusion (45s)
- Near-zero d_z (0.07 and −0.04)
- Floor effect: short, factual, single-page summaries leave little semantic room
- LME singular-fit for these models → low within-user variance
- Doesn't mean no reasoning shift — means our vector-similarity probe can't see it

## Slide 10 — Null is Method-Robust: LIWC Replication (45s)
- Reran CT/confusion with LIWC dictionary instead of word vectors
- LIWC Insight, CogMech (CT analogs) and Tentat, Discrep (confusion analogs)
- All |d_z| < 0.16 in STAIRS, same null story
- Rules out word-vector similarity as the reason for the null — floor effect is real

## Slide 11 — Does Dialogue Engagement Matter? (1 min) ⭐
- Recovered both sides of STAIRS dialogue from raw chat logs (original extraction kept only bot turns)
- Median student: 1 turn, 12 words — most "dialogues" are a single terse exchange
- Tertile split on total student words spoken to STAIRS:
  - low [0–6 words, n=54], medium [7–17, n=51], high [18–93, n=52]
- **Figure:** `figs/engagement_tiers_dz.png`
- **Clean monotone dose-response on structural/content features:**
  - word_count d_z: 0.68 → 1.07 → 1.13 (control = 0.34)
  - content_score d_z: 0.71 → 0.98 → 1.17 (control = 0.25)
- Confirmed by continuous moderator: ρ(log user words, word_count Δ) = +0.25, p = .001
- CT/confusion null holds across *every* tier — floor effect isn't a selection artifact

## Slide 12 — Takeaways, Limitations, Next Steps (1.5 min)
- **Takeaway 1:** STAIRS reliably produces longer, more content-complete revisions without sacrificing lexical quality
- **Takeaway 2:** Effect scales with dialogic engagement — students who give STAIRS a substantive answer revise the most (d_z ≈ 1.1 on content)
- **Takeaway 3:** Reasoning-level effects are undetectable with both vector and dictionary instruments — the null is robust, not an artifact
- **Limitations:** single domain (economics), single textbook, immediate-only outcome, feedback timing inferred from timestamps; engagement stratification is within-STAIRS (no control analog) so dose-response mixes treatment with user motivation
- **Next:** richer NLP (dependency/rhetorical parsing) or human coding for reasoning; cross-domain replication; downstream retention / exam outcomes; qualitative discourse analysis of feedback-revision loops
