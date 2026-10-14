# FreightTiger case study: plain English study guide

This guide explains the FreightTiger case study in simple, everyday language. Use this to prepare for your 10-minute walkthrough in Round 2.

---

## 1. The problem in simple terms

FreightTiger is a logistics platform. Companies hire trucks through FreightTiger to move goods between cities (such as Mumbai to Pune, or Delhi to Jaipur).

Over time, shipping costs quietly go up.
- Sometimes there is a good reason: diesel prices rose nationwide, floods forced trucks onto longer detours, or a festival created a shortage of trucks.
- Sometimes there is no good reason: transporters simply raise their rates.

Your job was to build a Python tool that:
1. Calculates shipping cost per tonne-km for each route, week by week.
2. Flags weeks where costs jumped abnormally compared to the route's own past and compared to similar routes.
3. Checks 10 short news notes to see if there is an honest explanation.
4. Writes a clear verdict: `No (justified)` if a real note explains it, or `Yes` if it is unexplained.

---

## 2. What files were provided

All files were located in `AI Intern Case Study/`:

1. **`shipment_records.csv`** (2,940 rows):
   - Shipments between January 2024 and December 2025 across 7 corridors:
     - Short routes: Delhi-Jaipur, Mumbai-Pune
     - Medium routes: Ahmedabad-Mumbai, Chennai-Bangalore, Kolkata-Bhubaneswar
     - Long routes: Delhi-Chennai, Mumbai-Delhi
2. **`context_notes.csv`** (10 notes):
   - Short notes on weather, fuel prices, tolls, and festivals.
3. **`sample_output_format_v2.csv`** (3 sample rows):
   - The format contract showing exact column names, numbers, and text style.
4. **`FreightTiger_Intern_CaseStudy.pdf`**:
   - The challenge brief and grading rules.

---

## 3. How the math works

### Metric: Cost per tonne-km
Raw rupee cost can mislead you if a truck carries 20 tonnes instead of 10 tonnes, or drives 500 km instead of 100 km. That is why freight uses cost per tonne-km:

$$\text{Cost per tonne-km} = \frac{\text{Total freight cost}}{\text{Total tonnes carried} \times \text{Total distance in km}}$$

**Example:**
If 3 trucks carried a combined 50 tonnes over 200 km (giving 10,000 tonne-km) and the total cost was ₹40,000:
$$\text{Cost per tonne-km} = \frac{40,000}{10,000} = 4.00 \text{ INR per tonne-km}$$

Important point: We sum all freight costs and divide by total tonne-km for the week. We do not calculate each truck's ratio and take an average. Doing that would let one tiny 1-tonne pickup distort the whole week's numbers.

---

### The two comparison baselines

For every route and every week (Monday to Sunday), we calculate two baselines:

#### Baseline 1: vs. own history
- What it does: compares this week to what this route cost in the past.
- The rule: average the cost per tonne-km over the trailing 8 weeks strictly before this week.
- No lookahead: future weeks are never used.
- Early weeks: week 1 has no prior history (`N/A`). In week 3, you have only 2 prior weeks, so you average those 2. You do not invent placeholder numbers.
- Formula:
  $$\text{Percent difference} = \frac{\text{Current cost} - \text{Past 8-week average}}{\text{Past 8-week average}} \times 100$$

#### Baseline 2: vs. similar routes (peers)
- What it does: compares this week to other routes of the same length category (Short, Medium, or Long) in the exact same week.
- The rule: average other routes in the same category.
- Peer exclusion: if calculating for Delhi-Jaipur (Short), its peer is Mumbai-Pune. Delhi-Jaipur is not included in its own peer average.
- Formula:
  $$\text{Percent difference} = \frac{\text{Current cost} - \text{Peer average}}{\text{Peer average}} \times 100$$

---

## 4. The evaluator traps in the notes

The 10 notes in `context_notes.csv` contain deliberate traps to test whether your system blindly grabs words or actually checks the facts:

| Note ID | What it says | Valid cost justification? | Explanation |
| :--- | :--- | :--- | :--- |
| N001 | Flooding on Chennai-Bangalore highway (Feb 24 to Mar 8) forced detours. | Yes | Real disruption, longer routes, higher trip costs. |
| N002 | Festival week on Ahmedabad-Mumbai created truck shortage and surcharges. | Yes | Real demand surge and rate increase. |
| N003 | Diesel prices rose nationwide starting May 5, pushing costs 5-7%. | Yes | Real systemic cost increase across all corridors. |
| N004 | New toll plaza opened, but affected routes are not part of this dataset. | No (Trap) | The note itself says it does not apply to our routes. |
| N005 | Mumbai-Delhi maintenance caused minor delays, costs were not significantly affected. | No (Trap) | The note explicitly says costs did not change. |
| N006 | Logistics report noted stable demand with no major disruptions. | No (Trap) | Normal demand cannot explain a price surge. |
| N007 | Delhi-Jaipur road resurfacing completed. | No (Trap) | Road improvements do not explain cost increases. |
| N008 | Normal freight movement on Kolkata-Bhubaneswar. | No (Trap) | Normal operations cannot explain a spike. |
| N009 | Chennai-Bangalore returned to normal after flood repairs. | No (Trap) | Returning to normal is not a reason for high costs. |
| N010 | Vehicle tracking mandate, costs absorbed by transporters without rate change. | No (Trap) | Transporters absorbed costs, so rates stayed unchanged. |

### The unexplained rule
If a cost spike occurs and no valid note justifies it:
- `flagged = Yes`
- `matched_note_id = ""` (blank)
- `reason = No matching note found... Cost rise looks unexplained and worth a human review.`

An honest unexplained answer scores higher than a confident hallucination.

---

## 5. How the codebase is structured

The code lives in `src/` and `tests/`:

- `src/core_analysis.py`: Contains the pure Pandas and NumPy math. It loads shipments, groups them into Monday-aligned weeks, computes cost per tonne-km, calculates both baselines, and flags candidate anomalies.
- `src/retrieval.py`: Uses TF-IDF sparse vector search and cosine similarity to retrieve top-k candidate notes for each corridor anomaly.
- `src/justification.py`: Enforces the deterministic verification gate (corridor scope, date boundaries, disruption polarity), filters out planted traps, and builds grounded reasons.
- `src/pipeline.py`: Runs the whole sequence, exports the clean CSV, and provides the plain English question-answering tool.
- `src/cli.py`: Gives you simple terminal commands to run the pipeline, verify reproducibility, and check costs.
- `src/eval_harness.py`: Evaluates the pipeline against a benchmark labeled set, calculating precision and recall.
- `tests/`: 13 automated unit and guardrail tests covering baselines, contracts, distractors, date boundaries, and retrieval sensitivity.

---

## 6. Three hand-computed spot checks

If an interviewer asks how you verified your math, show them these 3 checks:

1. **Delhi-Jaipur (week of 2024-11-11):**
   - Total freight cost: ₹214,142
   - Total tonne-km: 51,316.59
   - Cost per tonne-km: $214142 / 51316.59 = 4.1729 \rightarrow$ **4.17**
   - Trailing 8-week average was ₹3.0792 $\rightarrow$ **+35.5% vs own history**
   - Peer route (Mumbai-Pune) cost ₹3.4477 $\rightarrow$ **+21.0% vs similar routes**
   - Matched note: none $\rightarrow$ **Yes (unexplained)**

2. **Ahmedabad-Mumbai (week of 2025-01-20):**
   - Cost per tonne-km: **3.29**
   - vs. own history: **+29.5%**
   - vs. similar routes: **+22.5%**
   - Matched note: **N002** (festival surcharge) $\rightarrow$ **No (justified)**

3. **Mumbai-Pune (week of 2025-09-15):**
   - Cost per tonne-km: **3.98**
   - vs. own history: **+9.2%**
   - vs. similar routes: **+23.6%**
   - Closest note was N006, but N006 says demand was stable $\rightarrow$ **Yes (unexplained)**

Every number matches the sample file provided by FreightTiger.

---

## 7. 10-minute walkthrough cheat sheet

When you present in Round 2, follow these 5 steps:

1. **Start with the goal (1 min):**
   "I built a two-stage assistant. Stage 1 is deterministic data analysis that calculates weekly cost per tonne-km and spots cost jumps against history and peers. Stage 2 evaluates context notes with TF-IDF sparse vector retrieval followed by a strict verification gate to prevent false justifications."
2. **Explain the math (2 mins):**
   "To calculate cost per tonne-km, I summed all freight costs and divided by total tonne-km for the week. This prevents small shipments from distorting the route average. For own history, I used the trailing 8 weeks strictly prior to the current week so there is zero lookahead bias."
3. **Walk through the traps (2 mins):**
   "Several context notes were distractors. Note N004 was for routes outside our dataset, while notes like N005 and N006 noted that costs or demand were unaffected. My system rejects these. If no legitimate note applies, it flags the row as unexplained instead of guessing."
4. **Discuss determinism, architecture, and cost (2 mins):**
   "The evaluation required identical outputs across 3 runs. In financial and logistics auditing, anomaly verdicts must be repeatable, auditable, and legally defensible. Relying on an external LLM for simple date overlap checks introduces temperature drift, rate-limit crashes, and hallucination risks on planted distractor notes. I deliberately kept the core audit pipeline deterministic (Python/Pandas/TF-IDF) with $0.00 cost, zero API dependencies, and 100% byte-stable SHA-256 reproducibility. Generative AI belongs on the interactive query interface, not in the core financial ledger."
5. **Show the live demo (2-3 mins):**
   Run these live commands on screen:
   ```bash
   pytest tests/ -v
   python3 src/eval_harness.py
   python3 src/cli.py verify-reproducibility
   ```
   Show that all 13 unit tests pass, the evaluation harness scores 100%, and the SHA-256 hashes match.
