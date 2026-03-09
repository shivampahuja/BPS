---
title: CII - Consumer Interaction Intelligence
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app/app.py
pinned: true
license: mit
short_description: "Chattermill-style CII: topics, sentiment, behaviors, compliance, NPS joins"
tags:
  - nlp
  - sentiment-analysis
  - topic-modeling
  - consumer-intelligence
  - multilingual
  - bertopic
  - whisper
  - gradio
---

# Consumer Interaction Intelligence (CII) Platform

A production-ready, Chattermill-style **Consumer Interaction Intelligence** web application built with Gradio on Hugging Face Spaces. Ingests, unifies, analyzes, and visualizes consumer feedback across channels — calls, chats, emails, reviews, NPS verbatims — and converts it into decision-ready insights.

---

## What CII Adds (vs. Traditional TM/NPS-Only Views)

| Capability | Traditional | CII |
|---|---|---|
| Coverage | ~1% manual samples | Near-100% automated |
| Speed | Weeks | Real-time/daily |
| Behaviors | Not tracked | Empathy, auth, escalation, FCR |
| Compliance | Manual audit | Automated flagging |
| Actionability | Descriptive | CAPA + automation backlog |

### Biggest Business Questions Answered

1. **What topics/behaviors drive detractors vs promoters?** (by channel & market)
2. **Where do SOP/quality breakdowns occur most frequently?** (authentication, disclosures, escalation)
3. **Which issues are emerging/spiking?** (intervene before they hit NPS)
4. **What is the efficiency impact of Digital-First & VCA on contact reasons?** (automation eligibility)
5. **How do product issues translate into replacements and costs?** (CII + CSC/Product Care analytics)

---

## Features

- **Multi-channel Ingestion**: CSV/JSON uploads + WAV/MP3 call transcription (Whisper)
- **Privacy-First**: Microsoft Presidio PII redaction with audit logs; all analytics on masked text
- **PMI/CSC Taxonomy**: 7 domains → 25+ themes → 100+ sub-topics; guided BERTopic with seed words
- **Multilingual**: EN + CZ baseline; cross-lingual embeddings; language auto-detect
- **Sentiment & DSAT**: Multilingual sentiment + DSAT flagging with language-specific overlays
- **Behaviors & Compliance**: Empathy, authentication, resolution confirmation, escalation, adverse events
- **Anomaly Detection**: Rolling z-score/EWM on theme volumes, sentiment, KPIs; alert playbooks
- **NPS Join View**: Topic NPS, Topic Sentiment, CES links via shared TopicID
- **Impact Analysis**: Theme/behavior contribution to NPS/CSAT/cost via regularized models
- **Semantic Search & RAG**: Cross-lingual FAISS search; Haystack Q&A with interaction ID citations
- **Exports**: CSV/JSON reports, automation backlog, CAPA lists

---

## Architecture

```
/app
  app.py                    # Gradio entry: routes tabs/callbacks
  config.yaml               # models, languages, thresholds
  pipelines/
    ingest.py               # CSV/JSON/audio ingestion
    pii.py                  # Presidio PII redaction
    asr.py                  # Whisper transcription
    embed.py                # Sentence-Transformers embeddings
    topics.py               # BERTopic with guided seeds
    sentiment.py            # Multilingual sentiment + DSAT
    behaviors.py            # Behavior pattern detection
    compliance.py           # Compliance flag detection
    anomalies.py            # Anomaly/trend detection
    impact.py               # NPS/cost impact analysis
    rag.py                  # Semantic search + Haystack Q&A
  models/
    bertopic_model/         # Saved BERTopic artifacts
    vector_store/           # FAISS index
  data/
    samples/                # Demo CSVs + audio (EN, CS)
    taxonomy/pmc_taxonomy.yaml
  ui/
    components.py           # Reusable Gradio blocks
    layout.py               # Page layouts
  exports/
  tests/
i18n/en.json, i18n/cs.json
README.md
requirements.txt
```

---

## PMI/CSC Taxonomy (7 Domains)

| Domain | Key Themes |
|---|---|
| A. Product Experience | Device Performance, Consumables, Accessories, Replacement/Warranty |
| B. Service & Support (CSC) | Contact Handling Quality, Resolution Effectiveness, SOP & Compliance, Proactive Outreach |
| C. Digital Experience | Navigation & Usability, Account & Identity, Content & Help, Promotions & Pricing UX |
| D. Commerce & Fulfillment | Inventory & OOS, Delivery & Logistics, Payment & Refund, Order Management |
| E. Program & Loyalty | Onboarding/Guided Trial, Trade-in/Subscription, MGM/Rewards |
| F. Channel Experience | In-store Service, POS Execution, After-sales in Channel |
| G. Policy & Compliance | Age/Nicotine declarations, Privacy/PII, Health/Adverse events |

---

## Data Governance

- Responsible AI per PMI 44-C and Confidentiality Classification
- Privacy-by-default: PII redacted before any analytics
- Taxonomy versioning with audit trail
- Role-based access via HF Spaces secrets

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app/app.py
```

---

## Evaluation Criteria

- **ASR WER**: Reported for EN and CS samples
- **Topics**: Coherence & separation; merge/split tools available
- **Sentiment/DSAT**: Precision/recall on labeled subsets per language
- **Behaviors/Compliance**: Precision/recall on labeled subsets
- **Anomalies**: ≥80% spike detection, ≤20% false positive rate
- **Latency**: Upload→insight <10 min for 10k rows on CPU
- **Privacy**: 100% masking on configured entities
