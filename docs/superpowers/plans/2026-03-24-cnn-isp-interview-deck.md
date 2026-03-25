# CNN ISP Interview Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained HTML presentation for a 15-minute interview defense of the CNN-based ISP debugging demo, focused on engineering delivery, explainability, experiment evidence, challenges, and next steps.

**Architecture:** The deck will be generated as one standalone HTML file following the frontend-slides constraints. Existing project outputs and UI artifacts will be reused where possible, and any missing evidence will be generated from the local FastAPI demo so the presentation shows real system outputs instead of placeholder mockups.

**Tech Stack:** HTML, inline CSS/JS, frontend-slides viewport rules, Python/FastAPI demo assets, local project outputs

---

### Task 1: Lock Narrative And Asset Inventory

**Files:**
- Create: `docs/superpowers/plans/2026-03-24-cnn-isp-interview-deck.md`
- Create: `docs/presentation-assets/`
- Modify: `README.md` (reference only, no edits expected)

- [ ] **Step 1: Review the existing README, UI, and outputs**

Run: `sed -n '1,260p' /Users/hedyliang/Desktop/cnn_iqa_demo/README.md`
Expected: clear project scope, architecture, metrics, and output schema for slide copy.

- [ ] **Step 2: Enumerate reusable evidence**

Run: `ls -R /Users/hedyliang/Desktop/cnn_iqa_demo/outputs`
Expected: overlay, mask, and structure images available for results slides.

- [ ] **Step 3: Freeze the slide outline**

Expected outline:
1. Cover
2. Problem and motivation
3. Goal and scope
4. System architecture
5. End-to-end pipeline
6. Semantic + structure analysis design
7. Scoring and suggestion generation
8. Engineering implementation
9. Experiment evidence I
10. Experiment evidence II
11. Difficulties and trade-offs
12. Future work and Q&A

- [ ] **Step 4: Record the visual direction**

Expected: choose a professional, high-contrast preset suitable for interview delivery, with restrained motion and strong hierarchy.

### Task 2: Produce Presentation Assets

**Files:**
- Create: `docs/presentation-assets/`
- Create: `docs/presentation-assets/*.png`
- Modify: `outputs/` (read existing files only unless new demo outputs are generated)

- [ ] **Step 1: Reuse existing output images where sufficient**

Run: `ls /Users/hedyliang/Desktop/cnn_iqa_demo/outputs | sort`
Expected: at least one semantic overlay group and one structure overlay group suitable for slides.

- [ ] **Step 2: Generate fresh demo outputs only if evidence is missing**

Run: `python /Users/hedyliang/Desktop/cnn_iqa_demo/main.py` or `uvicorn api:app --reload`
Expected: local demo available at `http://127.0.0.1:8000`.

- [ ] **Step 3: Capture representative screenshots for the interview deck**

Expected assets:
- UI overview screenshot
- Semantic overlay/mask screenshot
- Structure overlay/mask screenshot
- Scoring/suggestions screenshot or cropped result panel

- [ ] **Step 4: Store selected artifacts under `docs/presentation-assets/`**

Expected: stable slide assets decoupled from transient run outputs.

### Task 3: Build The Standalone Slide Deck

**Files:**
- Create: `docs/cnn-isp-interview-deck.html`
- Modify: `/Users/hedyliang/.codex/skills/frontend-slides/viewport-base.css` (reference only, no edits expected)

- [ ] **Step 1: Read the frontend-slides supporting references**

Run:
- `sed -n '1,260p' /Users/hedyliang/.codex/skills/frontend-slides/html-template.md`
- `sed -n '1,260p' /Users/hedyliang/.codex/skills/frontend-slides/viewport-base.css`
- `sed -n '1,260p' /Users/hedyliang/.codex/skills/frontend-slides/animation-patterns.md`

Expected: mandatory viewport CSS and navigation behavior understood before authoring.

- [ ] **Step 2: Write the HTML presentation**

Expected structure:
- single self-contained HTML file
- inline CSS and JS
- no build tools
- 12 slides max
- full `viewport-base.css` included
- keyboard, wheel, touch, progress, and nav-dot support

- [ ] **Step 3: Keep each slide within density limits**

Expected: no scrolling inside slides, no overloaded bullet lists, experiment content split across multiple slides if needed.

- [ ] **Step 4: Use real project details in speaker-facing copy**

Expected content emphasis:
- ISP tuning pain points
- why semantic regions matter
- why structure partitions improve interpretability
- how FastAPI + debug dashboard make the system demonstrable
- what engineering trade-offs were made

### Task 4: Verify The Deck

**Files:**
- Modify: `docs/cnn-isp-interview-deck.html`

- [ ] **Step 1: Open the deck locally**

Run: `open /Users/hedyliang/Desktop/cnn_iqa_demo/docs/cnn-isp-interview-deck.html`
Expected: the browser opens the deck.

- [ ] **Step 2: Validate navigation and animation behavior**

Expected:
- arrow keys and space move slides
- nav dots work
- progress bar updates
- reduced-motion fallback does not break layout

- [ ] **Step 3: Validate viewport fit**

Expected:
- each slide fits at standard desktop size
- no internal slide scrolling
- image slides stay within `max-height: min(50vh, 400px)`

- [ ] **Step 4: Final review for interview framing**

Expected:
- presentation tone favors engineering delivery over academic theory
- challenge and future-plan slides are specific, not generic
- wording is concise enough for a 15-minute talk
