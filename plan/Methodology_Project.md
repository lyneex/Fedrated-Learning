# System Design and Methodology

FedRL-ClinicalDSS: A Privacy-Preserving Federated Reinforcement-Learning Framework for

Multimodal Clinical Decision Support

Author Name(s)

Affiliation / Department

October 8, 2026

Prepared in Research-Paper Format


## Contents


## 1 Introduction to System Design

The proposed system, FedRL-ClinicalDSS (Federated Reinforcement Learning-based Clinical Decision Support System), is a privacy-preserving, multi-institutional frame- work for collaborative clinical decision-making. It leverages multimodal patient data— Electronic Health Records (EHR), physiological vitals, and medical imaging—processed through modality-specific local encoders, and aggregated via a federated reinforcement learning (RL) policy head. The architecture ensures that no raw patient data leaves the originating hospital, while enabling the collective intelligence of a distributed hospital network to improve diagnostic and therapeutic suggestions over time.

This section presents the system architecture, component design, data and control flow, the federated learning and reinforcement learning methodology, authenticity and robustness mechanisms, and the clinician-in-the-loop feedback pipeline—all formulated to satisfy the dual objectives of clinical utility and regulatory compliance.

## 1.1 Motivation

- Single-hospital models generalise poorly; pooling knowledge improves accuracy and fairness.

- Legal and ethical constraints prevent raw data centralisation; federated approaches preserve locality.

- Human-in-the-loop reinforcement learning lets the system learn from real clinician judgments, increasing clinical relevance and trust.

- Inaccurate local datasets, broken ETL pipelines, or biased feedback can harm learning— systematic authenticity checks and quarantine workflows are therefore required.

## 2 System Architecture Overview

The architecture follows a hub-and-spoke federated topology comprising two prin-

cipal tiers (Table 1). [URL 🔗](#page-0)

## 3 Methodology

The system methodology integrates three intersecting paradigms:

- (a) Multimodal Deep Learning for patient representation,

- (b) Federated Learning (FL) for privacy-preserving collaboration, and


*Table 1: System tiers and their roles.*

| Tier | Component | Role |
| --- | --- | --- |
| Edge (Spoke) Hospital Client Node Local data ingestion, pre-processing, en- |   |   |
|   |   | coding, local RL training, clinician feed- |
|   |   | back capture, and benchmark evaluation |
| Central (Hub) Aggregation Server Secure aggregation of policy updates, ro- |   |   |
|   |   | bust filtering, global model validation, |
|   |   | rollback management, and model registry |

## Central Aggregation Server

*Figure 1: High-level hub-and-spoke architecture of FedRL-ClinicalDSS. Solid arrows represent encrypted local updates sent to the server; dashed arrows represent the broad- cast of the updated global policy.*


- (c) Human-in-the-Loop Reinforcement Learning (HITL-RL) for continuous policy improvement.

The overall process follows an iterative, epoch-based cycle aligned with fixed-period data snapshots (e.g. 1–2 months).

## 3.1 Methodological Framework

The methodology is structured into six sequential phases per federated round (Fig-

ure 2). [URL 🔗](#page-0)

*Figure 2: Six-phase iterative methodology per federated round.*

## 4 Detailed Component Design

## 4.1 Phase 1: Local Data Ingestion, Preprocessing, and Quality Assurance

Each hospital client operates an automated ETL (Extract, Transform, Load) pipeline on fixed-period data snapshots.

## 4.1.1 Data Sources and Modalities

## 4.1.2 Automated Pre-Training Quality Checks

Before any model training begins on a snapshot the following checks execute automati- cally:

- 1. Schema Validation: Verify column names, data types, and required-field presence against a canonical schema.

- 2. Unit Normalisation: Detect and convert non-standard measurement units (e.g. Fahrenheit Celsius).

- 3. Range Checks: Flag physiologically implausible values (e.g. heart rate > 300 bpm, age < 0).

- 4. Deduplication: Identify and remove duplicate patient encounters using composite keys.


*Table 2: Data modalities, sources, and preprocessing steps.*

| Modality Source Examples Format Preprocessing |   |   |   |   |
| --- | --- | --- | --- | --- |
| Tabular | Demographics, | Structured | Schema validation, unit |   |
| EHR | ICD codes, medi- | CSV / FHIR | normalisation, | missing- |
|   | cations, lab results |   | value imputation (me- |   |
|   |   |   | dian/mode), one-hot / |   |
|   |   |   | embedding encoding |   |
| Time-Series | Heart rate, SpO2, | Temporal se- | Resampling to fixed fre- |   |
| Vitals | blood pressure, res- | quences | quency, z-score normalisa- |   |
|   | piratory rate |   | tion per channel, window- |   |
|   |   |   | ing, outlier capping |   |
| Medical | Chest X-rays, CT | DICOM | Metadata stripping, resiz- |   |
| Imaging | slices, fundoscopy | PNG / NIfTI | ing (224×224), intensity nor- |   |
|   |   |   | malisation, augmentation |   |

- 5. Label Heuristics: Cross-reference diagnostic labels against coded lab results and imaging findings.

- 6. Snapshot Hashing: Compute SHA-256 hash of the finalised dataset and log to an immutable provenance ledger.

Algorithm 1 Local Data Quality Assurance Pipeline

Require: Raw snapshot Draw, Canonical schema Range configuration

Ensure: Validated snapshot Dvalid, QA report Q, Hash h

- 1: 2: 3: 4: 5: Extract(Draw)

- SchemaValidate(D,

- UnitNormalise(D)

- RangeCheck(D,

- Deduplicate(D)

- 6: Q LabelHeuristicCheck(D)

- 7: if Q.critical_failures > 0 then

- 8: Flag site; Halt training; Notify administrator

- 9: end if

- 10: Dvalid

- 11: h SHA256(Dvalid)

- 12: Log(h, timestamp, site_id) to provenance ledger

- 13: return Dvalid, Q, h


## 4.2 Phase 2: Multimodal Encoding and Fusion

## 4.2.1 Modality-Specific Encoders

Three independent encoder networks extract fixed-dimensional embedding vectors from each data modality.

## (a) Tabular / EHR Encoder (fehr).

- Architecture: Multi-Layer Perceptron (MLP) with optional longitudinal transformer attention for sequential visit data.

- Input: Pre-processed feature vector xehr Rdehr.

- Output: zehr = fehr(xehr) R256.

- Rationale: MLPs efficiently handle heterogeneous tabular features; transformer layers capture temporal dependencies across visits.

## (b) Time-Series Vitals Encoder (fvitals).

- Architecture: 1-D Convolutional Neural Network (1D-CNN) with residual blocks, or bidirectional GRU/LSTM.

- Input: Multivariate time-series Xvitals RT×C (T = time steps, C = channels).

- Output: zvitals = fvitals(Xvitals) R256 (via global average pooling or final hidden state).

- Rationale: 1D-CNNs offer efficient temporal pattern extraction with lower computa- tional cost than full RNNs.

- (c) Medical Imaging Encoder (fimg).

- Architecture: Lightweight CNN (e.g. EfficientNet-B0) or compact Vision Transformer (ViT-Tiny), pre-trained on ImageNet and fine-tuned.

- Input: Pre-processed image ximg RH×W×C.

- Output: zimg = fimg(ximg) R256.

- Rationale: Transfer learning from ImageNet provides strong visual-feature extraction; lightweight architectures respect hospital compute constraints.


## 4.2.2 Multimodal Fusion Layer

Embeddings from available modalities are fused into a unified patient representation. Let

m = [mehr, mvitals, mimg]

modality k, we set zk 0 R256 and mk 0. The fused representation is:

An optional linear projection reduces dimensionality:

1}3 denote the modality-presence mask. For any missing

The modality-presence mask explicitly informs the policy head about which data sources contributed to the representation, enabling the model to learn modality-specific reliability weighting.

*Figure 3: Multimodal encoding and fusion pipeline. Three modality-specific encoders produce 256-dimensional embeddings that are concatenated with a modality-presence mask, layer-normalised, and projected to a 512-dimensional patient representation.*

## 4.3 Phase 3: Local Reinforcement Learning Training

## 4.3.1 Problem Formulation as a Markov Decision Process

The clinical decision-support task is formulated as a contextual bandit or MDP (Table 3). [URL 🔗](#page-0)

*Table 3: MDP formulation of the clinical decision-support task.*

| Element | Definition |   |
| --- | --- | --- |
| State (s) | zpatient | R512—the fused multimodal embedding |
| Action (a) | Suggested diagnosis, treatment plan, or ranked list |   |
|   | of clinical actions from discrete action space |   |
| Reward (r) Numeric signal derived from clinician feedback |   |   |
|   | (§4.4) |   |
| Transition (s′) Next patient encounter or updated patient state |   |   |
| Policy (π) | πθ(a | s)—parameterised by the policy-head net- |
|   | work θ |   |


## 4.3.2 Policy Head Architecture

The policy head follows an Actor-Critic design:

Vϕ(s) = FC128→1 (︂ ReLU(FC256→128(ReLU(FC512→256(s)))) )︂

The combined loss function is:

Lactor ⏟⏟

Lcritic ⏟⏟

Entropy bonus ⏟⏟

where Aˆ = r −Vϕ(s) is the advantage estimate, and c1, c2 are weighting coefficients.

Starting Configuration. A contextual bandit (single-step decisions) is used for initial deployment where feedback is sparse. The system upgrades to full Advantage Actor- Critic (A2C) or Proximal Policy Optimisation (PPO) when sufficient trajectory data and sequential decision scenarios emerge.

## 4.3.3 Local Training Algorithm

Algorithm 2 Local RL Training per Federated Round

Require: Validated snapshot Dvalid, global policy θglobal,

Ensure: Policy delta ∆θ

2:

Fuse (︂ fehr(xehr), fvitals(Xvitals), fimg(ximg), m )︂

3:

4:

5:

6: for epoch = 1 to Elocal do

7: Sample mini-batch {(si, ai, ri)}

B

Aˆi

8:

9:

10:

11:

12:

13:

14: ∆θ

15:

16: Encrypt(∆θ) via secure aggregation protocol

17:

replay buffer feedback log

B,

for each encounter (x, aprev, r) in Dvalid

do

UF

zpatient

Store (zpatient, aprev, r) in

end for

from

ri −Vϕ(si)

Advantage

Compute Ltotal via Eq. (5) [URL 🔗](#page-0)

θlocal

θlocal

η ∇θLtotal

ϕlocal

ϕlocal

η ∇ϕLcritic

Optional encoder fine-tuning

ηenc ∇fkLtotal

end for

θlocal

θglobal

Clip(∆θ, max_norm = γ)

Norm clipping

return ∆θ


## 4.4 Phase 3a: Clinician-in-the-Loop Feedback and Reward De- sign

## 4.4.1 Feedback Collection Workflow

*Figure 4: Clinician feedback collection workflow.*

## 4.4.2 Reward Mapping Function

*Table 4: Mapping from clinician actions to numeric rewards.*

| Clinician Action Semantic Meaning |   | Reward (r) |
| --- | --- | --- |
| Confirm | Suggestion accepted as-is | +1.0 |
| Partial Accept | Suggestion modified slightly | +0.5 |
| Reject | Suggestion not useful | 0.0 |
| Harmful | Suggestion potentially dangerous | −1.0 |
| Correct a′ | Clinician provides correct action +0.5 / store (s, a′, +1.0) |   |

## 4.4.3 Reward Normalisation

To mitigate inter-site and inter-clinician bias:

where µsite and σsite are the running mean and standard deviation of rewards at site i, and ε = 10−8 ensures numerical stability.

## 4.4.4 Delayed Outcome Integration (Optional)

For conditions where ground truth emerges after days or weeks:

with default α = 0.7.


## 4.5 Phase 4: Federated Secure Aggregation

## 4.5.1 Aggregation Protocol

## Algorithm 3 Robust Federated RL Aggregation

```
Require: Encrypted deltas i=1, site weights thresholds τnorm, τcos
Ensure: Updated global policy θ(t+1)
global
— Step 1: Secure Decryption —
1: SecureDecrypt({∆θi}) Bonawitz et al. protocol
— Step 2: Per-Update Anomaly Detection —
2: for each site i do
3: ni
4: ci cos(∆θi, ∆θj̸=i)
5: if ni > τnorm or ci < τcos then
6: Flag site i; wi 0
7: end if
8: end for
— Step 3: Robust Aggregation —
9: Svalid : site i not flagged}
10: Option A: ∆θagg TrimmedMean({∆θi}i∈Svalid , β%)
11: Option B: ∆θagg CoordMedian({∆θi}i∈Svalid)
12: Option C: ∆θagg MultiKrum({∆θi}i∈Svalid , f=⌊N/4⌋)
13: Default (FedAvg): ∆θagg ∑︁ i∈Svalid ∑︁wi ∆θi
j wj
— Step 4: Global Update —
14: θ(t+1) global θ(t) global + ηglobal ∆θagg
15: return θ(t+1)
global
```

## 4.5.2 Privacy Mechanisms

## 4.6 Phase 5: Benchmarking, Authenticity Verification, and Quar- antine

## 4.6.1 Multi-Level Authenticity Check Framework

Four levels of checks are applied sequentially (Figure 5). [URL 🔗](#page-0)


*Table 5: Privacy mechanisms employed in FedRL-ClinicalDSS.*

| Mechanism | Implementation | Purpose |
| --- | --- | --- |
| Secure Aggregation Bonawitz et al. (2017) with |   | Server sees only aggre- |
|   | secret sharing | gated updates |
| Differential Privacy Client-side Gaussian noise |   | Formal privacy guaran- |
|   | (ε-DP) with per-sample | tees |
|   | gradient clipping |   |
| TLS Encryption End-to-end TLS 1.3 |   | Eavesdropping protec- |
|   |   | tion |
| Provenance Hash- | SHA-256 of each update | Tamper detection and |
| ing | payload | audit trail |

*Figure 5: Multi-level authenticity check pipeline.*

## Algorithm 4 Site Quarantine Decision

```
Require: Site check results = {level1, . . . , level4}
Ensure: Site status {Active, Warned, Quarantined}
1: severity 0
2: for each level L in do
3: if L.critical_fail then severity += 3
4: else if L.warning then severity += 1
5: end if
6: end for
7: if severity 5 then
8: return Quarantined Exclude from aggregation; notify governance
9: else if severity 2 then
10: return Warned Down-weight contribution by 0.5; schedule review
11: else
12: return Active
13: end if
```


## Algorithm 5 Global Policy Validation and Rollback

Require: Candidate policy θcand, previous policy θprev, holdout Dhold, safety suite Tsafe,

thresholds δperf, δsafe

Ensure: Deployed policy θdeployed

- 1: pnew Evaluate(θcand, Dhold)

- 2: pprev Evaluate(θprev, Dhold)

- 3: snew SafetyTest(θcand, Tsafe) 4: if snew < δsafe then

5:

6:

7:

8:

9:

10:

11:

- 12: Update model registry with version, timestamp, metrics 13: end if

14:

Safety rollback

θdeployed

θprev

Alert(“Safety regression”)

else if pnew < pprev

δperf then

Performance rollback

θdeployed

θprev

Alert(“Performance regression”)

else

θdeployed

θcand

return θdeployed

*Table 6: Safety mechanisms enforced during production deployment.*

| Mechanism | Description |
| --- | --- |
| Mandatory Clinician | All high-risk suggestions (e.g. medication changes, |
| Confirmation | surgical recommendations) require explicit clini- |
|   | cian approval before downstream action. |
| Conservative Explo- | ε-greedy with ε 0.05 in production; higher ex- |
| ration | ploration only in simulation. |
| Confidence Thresh- | Suggestions with V(s) < 0.3 are flagged as “low |
| olds | confidence.” |
| Action Blocklist | Known dangerous action combinations are hard- |
|   | blocked regardless of policy output. |


## 4.6.2 Quarantine and Remediation Workflow

## 4.7 Phase 6: Global Validation, Safety, and Rollback

## 4.7.1 Post-Aggregation Validation Pipeline

## 4.7.2 Safety Constraints in Production

## 5 End-to-End Data Flow

*Figure 6: End-to-end data flow in FedRL-ClinicalDSS. Red arrows indicate fail- ure/quarantine paths; dashed blue arrows indicate the global policy broadcast.*

## 6 Technology Stack Justification

## 7 Evaluation Methodology

## 7.1 Evaluation Metrics

## 7.2 Experimental Design

- 1. Baseline Comparison: Single-site models vs. centralised (oracle) vs. FedRL-ClinicalDSS.

## 2. Ablation Studies:

- (a) Without RL (supervised only).

- (b) Without robust aggregation.

- (c) Without multimodal fusion (single modality).


*Table 7: Technology choices and justifications.*

| Layer | Technology | Justification |
| --- | --- | --- |
| Deep Learning PyTorch |   | Dynamic graphs suit RL; |
|   |   | strong ecosystem (TorchRL, |
|   |   | PFL) |
| Federated Learning PySyft / Flower / |   | Mature FL frameworks with |
|   | NVIDIA FLARE | secure aggregation |
| RL Library | Stable-Baselines3 / | Tested PPO/A2C/bandit im- |
|   | TorchRL | plementations |
| Imaging Models | torchvision / timm Pre-trained weights; proven |   |
|   |   | clinical performance |
| Data Processing Pandas, NumPy, |   | Industry standard for tabu- |
|   | SciPy | lar/statistical ops |
| Communication gRPC + TLS 1.3 |   | High-performance, secure |
|   |   | inter-node comms |
| Provenance | MLflow / W&B + | Experiment tracking; audit |
|   | append-only logs | compliance |
| Database | PostgreSQL + | Relational metadata; object |
|   | MinIO/S3 | storage for artefacts |
| Clinician UI | React.js / Flask | Minimal-friction browser- |
|   |   | based interface |

*Table 8: Evaluation metrics and target values.*

| Category | Metric | Target |
| --- | --- | --- |
| Clinical Accuracy AUROC, AUPRC |   | > 0.85 vs. single-site |
|   |   | baseline |
| Top-K Accuracy Correct diagnosis in top-3 |   | > 80% |
| Calibration | Expected Calibration Error | < 0.05 |
|   | (ECE) |   |
| Feedback Efficiency Average reward per round Monotonically in- |   |   |
|   |   | creasing |
| Clinician Accep- | Suggestion acceptance rate | > 70% after conver- |
| tance |   | gence |
| Robustness | Byzantine-site detection | 90% |
|   | rate |   |
| Latency | Data suggestion time | < 5 s |
| Privacy | Formal DP guarantee (if en- | ε 8 per round IN |
|   | abled) |   |


- (d) Without clinician feedback.

- 3. Byzantine Robustness Testing: Inject 10–30% malicious sites with label-flipped, noise-injected, or model-poisoning attacks; measure policy degradation and detection rate.

- 4. Scalability Testing: 5, 10, 25, 50, and 100 simulated hospital clients; measure com- munication overhead and convergence rate.

## 8 Governance, Compliance, and Audit Framework

*Table 9: Governance and compliance components.*

| Component | Implementation |
| --- | --- |
| Data Provenance Immutable snapshot hashes, ETL logs, and data- |   |
|   | lineage records |
| Model Provenance Versioned model registry with full training metadata |   |
|   | per round |
| Feedback Prove- | Pseudonymised clinician ID, timestamp, policy ver- |
| nance | sion, action, reward |
| Incident Response Quarantine playbook, automated rollback, escalation |   |
|   | procedures |
| Onboarding Certi- | ETL validation, schema-adapter tests, and bench- |
| fication | mark evaluation before participation |
| Regulatory Align- | HIPAA (US), GDPR (EU), and relevant local health- |
| ment | care data-protection regulations |

## 9 Reinforcement Learning Design Choices

## 10 Robustness and Authenticity Defences

- 1. Local pre-checks and monthly snapshot evaluation against baselines (Algorithm 1). [URL 🔗](#page-0)

- 2. Update clipping, normalised weighting, and robust aggregation—trimmed mean, co- ordinate median, Multi-Krum (Algorithm 3). [URL 🔗](#page-0)

- 3. Per-update anomaly detection: gradient norms, cosine similarity, unusual reward dis- tributions.

- 4. Quarantine workflow with severity scoring and human-in-the-loop remediation (Algo- rithm 4). [URL 🔗](#page-0)

- 5. Trusted holdout validation and automatic rollback on degradation (Algorithm 5). [URL 🔗](#page-0)


*Table 10: Key RL design decisions and rationale.*

| Aspect | Choice and Rationale |
| --- | --- |
| Policy Type | Start with a contextual bandit for suggestion ranking |
|   | (sample-efficient when feedback is sparse); expand to |
|   | PPO/A2C for richer sequential decisions. |
| Reward Design | Simple numeric mapping (Table 4) with optional |
|   | delayed-reward integration (Eq. 7). Normalise re- |
|   | wards per clinician/site to reduce bias (Eq. 6). |
| Replay & Off- | Store local (s, a, r, s′) tuples; use importance sam- |
| Policy | pling when aggregating off-policy data across sites. |
| Exploration | Conservative in production (ε 0.05); prefer |
|   | safe/uncertain flags. Use simulated environments for |
|   | heavier exploration. |
| Privacy | Secure aggregation (Bonawitz-style) so server sees |
|   | only aggregated updates; optional client-side DP with |
|   | careful utility–privacy trade-off. |

## 11 Constraints and Risks

## 12 Deliverables and Components

- 1. Local Client Package: Modality encoders, concatenation fusion layer, local training loop, local RL updater, pre-check scripts, and feedback logger.

- 2. Central Aggregator: Secure aggregation module, robust policy aggregator, replay utilities, validation pipeline, model registry, and rollback system.

- 3. Clinician Feedback UI & Reward Mapping: Lightweight web-based review flows and documented reward rules.

- 4. Benchmark & QA Suite: Standardised benchmarks, simulated bad-data tests, and safety test-suite.

- 5. Governance Documentation: Onboarding checklist, quarantine playbook, incident- response procedures, and audit-log specifications.


*Table 11: Key constraints, associated risks, and mitigation strategies.*

| Constraint | Risk | Mitigation |
| --- | --- | --- |
| Sparse clinician | Insufficient reward signal | Careful reward shaping; |
| feedback | for policy improvement | replay buffers; contextual |
|   |   | bandits for sample effi- |
|   |   | ciency |
| Heavy DP noise Reduced RL learning sig- |   | Tune ε; use secure aggrega- |
|   | nal | tion as primary mechanism; |
|   |   | DP as optional layer |
| Population differ- | Legitimate sites flagged | Tune thresholds with hu- |
| ences | and quarantined | man review; allow per-site |
|   |   | personalisation layers |
| Operational burden Low clinician engage- |   | Minimal UX friction; inte- |
|   | ment reduces feedback | grate into existing workflow; |
|   | volume | gamification incentives |
| Adversarial partici- | Model poisoning or data | Multi-level authenticity |
| pants | poisoning attacks | checks; robust aggregation; |
|   |   | anomaly detection |

*Table 12: Phased implementation roadmap.*

| Phase Timeline Activities |
| --- |
| 1 Months 1–2 Build local encoder prototypes; choose embed- |
| ding size (256-D); implement ETL and pre- |
| check scripts. |
| 2 Months 2–3 Prototype local RL policy head (contextual |
| bandit / small actor-critic); build feedback log- |
| ger and minimal UI. |
| 3 Months 3–5 Build secure aggregator with FedAvg baseline |
| + trimmed-mean fallback; implement holdout |
| validation and rollback. |
| 4 Months 5–7 Run closed simulations with injected bad data |
| and biased feedback; tune thresholds and quar- |
| antine rules. |
| 5 Months 7–10 Pilot with 2–3 certified hospitals; iterate reward |
| mapping, UX, and robustness. |
| 6 Months 10–12 Scale to additional hospitals; comprehensive |
| evaluation and publication. |


## 13 Implementation Roadmap

## 14 Summary

The proposed FedRL-ClinicalDSS methodology integrates multimodal deep learning, federated learning, and human-in-the-loop reinforcement learning into a cohesive, privacy- preserving clinical decision-support framework. The six-phase iterative cycle—from local data ingestion and quality assurance, through multimodal encoding and fusion, local RL training with clinician reward signals, secure federated aggregation with robust defences, global validation with automatic rollback, to governed deployment—ensures that the system continuously improves its clinical recommendations while maintaining patient privacy, data authenticity, and operational safety.

The multi-level authenticity checking and quarantine mechanisms provide defence- in-depth against data-quality issues and adversarial participants, while the conservative exploration strategy and mandatory clinician confirmation for high-risk actions ensure patient safety remains paramount. Together, these design choices position FedRL- ClinicalDSS as a practical, scalable, and trustworthy framework for collaborative clin- ical intelligence.


## References

- [1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). Communication-efficient learning of deep networks from decentralized data. Proc. AISTATS.

- [2] Bonawitz, K., Ivanov, V., Kreuter, B., et al. (2017). Practical secure aggregation for privacy-preserving machine learning. Proc. ACM CCS.

- [3] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

- [4] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proc. MLSys.

- [5] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. Proc. NeurIPS.

- [6] Tan, M. & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proc. ICML.

- [7] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. Proc. ICLR.

- [8] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-resilient distributed learning: Towards optimal statistical rates. Proc. ICML.

- [9] Abadi, M., Chu, A., Goodfellow, I., et al. (2016). Deep learning with differential privacy. Proc. ACM CCS.

- [10] Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y., & Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. Journal of Clinical Epidemiology, 110, 12–22.
