# Orion — Coaching Rubric & Reference Sources

**Scope:** Scientific Foundation, Empirical Datasets, Industry Formulas, & Decision Rules for Orion's Guardrail Engine & AI Dataset Generator.  
**Status:** Working Reference Document.

---

## 1. Peer-Reviewed Sports Science Research

### 1.1 Acute:Chronic Workload Ratio (ACWR)
* **Citation:** Gabbett, T. J. (2016). *The training—injury prevention paradox: should athletes be training smarter and harder?* British Journal of Sports Medicine, 50(5), 273–280.
* **Formulas:**
  $$\text{ATL (Acute Training Load)} = \text{7-day rolling average TSS / Workload}$$
  $$\text{CTL (Chronic Training Load)} = \text{28-day rolling average TSS / Workload}$$
  $$\text{ACWR} = \frac{\text{ATL}}{\text{CTL}}$$
* **Threshold Rules:**
  * $\text{ACWR} < 0.8$: Under-training / De-conditioning hazard zone.
  * $0.8 \le \text{ACWR} \le 1.3$: **Optimal "Sweet Spot"** (low injury probability, positive adaptation).
  * $1.31 \le \text{ACWR} \le 1.49$: Elevated fatigue / caution zone.
  * $\text{ACWR} \ge 1.50$: **High Injury Risk "Danger Zone"** ($2\times - 4\times$ injury spike $\rightarrow$ mandatory volume cap / reduction).

---

### 1.2 Heart Rate Variability (HRV) Suppression
* **Citations:**
  * Plews, D. J., et al. (2012). *Heart rate variability in elite triathletes, is variation in variability the key to optimizing training adaptation?* International Journal of Sports Physiology and Performance.
  * Buchheit, M. (2014). *Monitoring training status with HR measures: do all roads lead to Rome?* Frontiers in Physiology.
* **Metric:** Nightly RMSSD (Root Mean Square of Successive Differences) in ms.
* **Baseline Calculation:** Rolling 7-day mean ($\mu_{7d}$) and standard deviation ($\sigma_{7d}$).
* **Threshold Rules:**
  * $\text{RMSSD}_{\text{today}} \ge \mu_{7d} - 0.5\sigma_{7d}$: Normal / Recovered.
  * $\mu_{7d} - 1.5\sigma_{7d} \le \text{RMSSD}_{\text{today}} < \mu_{7d} - 0.5\sigma_{7d}$: Mild Suppression ($>10\%$ drop) $\rightarrow$ swap high-intensity intervals to Zone 2.
  * $\text{RMSSD}_{\text{today}} < \mu_{7d} - 1.5\sigma_{7d}$ or $>15\%$ drop below $\mu_{7d}$ for 2+ days: **Severe Parasympathetic Suppression** $\rightarrow$ mandatory `REST_DAY` or active recovery.

---

### 1.3 Training Monotony & Strain Index
* **Citation:** Foster, C. (1998). *Monitoring training in athletes with reference to overtraining syndrome.* Medicine & Science in Sports & Exercise, 30(7), 1164–1168.
* **Formulas:**
  $$\text{Monotony} = \frac{\text{Mean Daily Load (7d)}}{\text{Standard Deviation Daily Load (7d)}}$$
  $$\text{Strain} = \text{Weekly Total Load} \times \text{Monotony}$$
* **Threshold Rules:**
  * $\text{Monotony} > 2.0$: High risk of adaptation plateau and upper respiratory tract infections (URTI) due to lack of variation $\rightarrow$ enforce rest day or high-low load distribution.
  * $\text{Strain} > \text{Athlete 28-day 90th percentile}$: Overtraining alert $\rightarrow$ enforce deload phase.

---

### 1.4 Chronic Sleep Restriction
* **Citation:** Milewski, M. D., et al. (2014). *Chronic lack of sleep is associated with increased sports injuries in adolescent athletes.* Journal of Pediatric Orthopaedics, 34(2), 129–133.
* **Threshold Rules:**
  * Total sleep $< 7.0 \text{ hours}$ for 2 consecutive nights: $1.7\times$ elevated injury risk.
  * Action: Disallow Anaerobic/Zone 5 workouts; cap workout intensity at Zone 2 / Tempo; enforce maximum 60-minute duration.

---

## 2. Empirical Research Datasets (Seed Inputs)

### 2.1 PMData (Simula Research Laboratory)
* **Description:** 5 months of continuous objective telemetry (Fitbit heart rate, sleep architecture, step counts) paired with PMSys subjective daily logs (fatigue, mood, soreness, stress, readiness) across 16 athletes.
* **Access URL:** [https://simula.no/PMData](https://simula.no/PMData) or Kaggle (`simularesearch/pmdata`).
* **Use Case:** Provides joint probability distributions of objective parameters (HRV, sleep) paired with subjective scores (1–5 / 1–10 scales) to generate realistic input states.

### 2.2 Wearable HRV & Sleep Diary Dataset (2025)
* **Description:** Open-access dataset of 49 participants tracking nightly HRV (RMSSD), resting HR, sleep architecture (deep/REM/light), and self-reported wellness logs over 4 weeks.
* **Access:** PhysioNet / Kaggle open access repositories.
* **Use Case:** Realistic nightly HRV variance, noise modeling, and baseline calibration distributions.

---

## 3. Industry Coaching Frameworks

### 3.1 TrainingPeaks / Andrew Coggan Methodology
* **Metrics:**
  * **TSS (Training Stress Score):** Measure of workout volume & intensity.
  * **ATL (Acute Training Load):** 7-day exponentially weighted TSS (Fatigue).
  * **CTL (Chronic Training Load):** 42-day exponentially weighted TSS (Fitness).
  * **TSB (Training Stress Balance):** $\text{TSB} = \text{CTL} - \text{ATL}$ (Form / Readiness).
* **Threshold Rules:**
  * $\text{TSB} < -30$: **Excessive Fatigue Hazard** $\rightarrow$ cap workout intensity to Zone 1/2.
  * $-10 \le \text{TSB} \le +10$: Neutral / Productive Training Range.
  * $+10 < \text{TSB} \le +25$: **Optimal Race Taper / Peak Performance Window**.
  * $\text{TSB} > +25$: De-conditioning / Prolonged Rest.

### 3.2 Garmin / Firstbeat Analytics Standards
* **EPOC (Excess Post-Exercise Oxygen Consumption):** Training Effect score (1.0–5.0 scale).
* **Stress Score:** Real-time HRV frequency-domain calculation (0–100). Stress $>75$ during sleep indicates incomplete autonomic recovery.
* **Body Battery:** Composite linear model combining HRV, stress, sleep quality, and daily movement.

---

## 4. Medical Safety Guidelines & Red Flags

### 4.1 ACSM (American College of Sports Medicine) Exercise Protocols
* **Hard Contraindications (Mandatory `ESCALATE_TO_HUMAN` Triggers):**
  1. **Systemic Infection / Fever:** Self-reported fever, body chills, or resting HR $>20\%$ above 30-day baseline $\rightarrow$ **Immediate `ESCALATE_TO_HUMAN`** (risk of viral myocarditis; absolute training stoppage).
  2. **Acute Localized Joint/Bone Pain:** Free-text reporting of sharp, non-muscle pain $\rightarrow$ **`ESCALATE_TO_HUMAN`** (risk of stress fracture or tendon rupture).
  3. **Unexplained Resting Tachycardia:** Resting HR elevated by $>15 \text{ bpm}$ for 2+ consecutive days without recent exercise $\rightarrow$ **`ESCALATE_TO_HUMAN`** (underlying illness / cardiac stress).
  4. **Unresolvable Signal Conflict:** Severe objective crash (HRV down 30%, sleep 4h) combined with athlete insisting on a maximal workout $\rightarrow$ **`ESCALATE_TO_HUMAN`** (cognitive dissonance / overreaching risk).

---

## 5. Decision & Escalation Matrix Summary

| Objective Input | Subjective Input | ACWR | Action Output | Rationale |
|---|---|---|---|---|
| Normal | Good (Fatigue $\le 4$) | $0.8 - 1.3$ | `MAINTAIN_PLAN` | Telemetry balanced; load within safe zone. |
| Mild HRV Drop ($10-15\%$) | Normal | $1.1 - 1.3$ | `SWAP_TO_ZONE2` | Autonomic fatigue detected; maintaining volume in Zone 2. |
| Severe HRV Drop ($>15\%$) | High Fatigue ($\ge 7$) | Any | `REST_DAY` | Parasympathetic suppression & subjective exhaustion. |
| Normal | Normal | $\ge 1.5$ | `MODIFY_INTERVALS` | Workload spike hazard ($\text{ACWR} \ge 1.5$); capping intensity. |
| Fever / Acute Sharp Pain | Any | Any | `ESCALATE_TO_HUMAN` | Red flag symptom detected; clinical/human coach evaluation required. |
