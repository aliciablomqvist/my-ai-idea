# ShelterMatch

Final project for the Building AI course

## Summary

ShelterMatch recommends adoptable pets to people based on **profile preferences** and **pet profiles (text + images)**. It uses embeddings and a learning-to-rank model to surface pets that best fit a household’s needs, helping shelters reduce length-of-stay and increase successful adoptions.


## Background

Millions of animals enter shelters each year. Adopters often browse randomly; shelters manually “match” based on experience. This is time-consuming and can overlook great fits (e.g., shy pets for quiet homes). An AI-assisted recommender can:
- Improve **discovery** for overlooked animals.
- Reduce **time-to-adoption** and **return rates** (better fit).
- Help staff prioritize outreach when a strong match appears.

## How is it used?

1. **Adopter profile** (5–10 quick questions): home type, yard, kids/other pets, activity level, grooming tolerance, allergies, size/age preferences, training expectations, location & radius.
2. **Inventory sync** from shelter APIs (e.g., Petfinder): pet species, breed mix, size, age, sex, compatibility flags, medical/behavior notes, photos.
3. **Ranking**: compute similarity between adopter needs and pet profiles (text + image features), then apply a learned re-ranker that incorporates adoption likelihood and shelter constraints (distance, special needs).
4. **Recommendations UI**: a web page for adopters with filters, reasons (“Loves quiet homes; low-shedding”), and contact/visit flow. A staff dashboard highlights priority matches.
5. **Feedback loop**: saves/likes, inquiries, and adoption outcomes refine the model.

_Users_: public adopters and shelter staff.  
_Context_: mobile-friendly web app.  

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300" height="500">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg" width="300" height="500">
</p>

## Data sources and AI methods
**Data**
- **Pet data**: Petfinder API (profiles, attributes, media, location, status).
- **Adopter data**: voluntary questionnaire; minimal PII (email for follow-up).
- **Auxiliary**: city/ZIP geocoding, shelter metadata (hours, policies).

**Features**
- **Text**: pet descriptions, tags → sentence embeddings (e.g., SBERT).
- **Images**: pet photos → vision embeddings (e.g., CLIP) + quality heuristics.
- **Structured**: species, age, size, distance, good-with-kids/pets, energy level.
- **Adopter**: activity level, home constraints, allergies, preferred size/age.

**Models**
- **Stage 1 (candidate retrieval)**: cosine similarity between adopter embedding and pet embeddings (text+image fusion, e.g., weighted sum) to get top-N (e.g., 100).
- **Stage 2 (re-ranking)**: Learning-to-Rank (e.g., LambdaMART / XGBoostRanker) on features such as distance, compatibility matches, historical adoption likelihood, photo quality, novelty/diversity.
- **Cold start**: rule-based fallback + popularity priors; quickly personalize via clicks/saves.

**Evaluation**
- Offline: **NDCG@k**, **MRR**, **Precision@k** on historical inquiry/adoption logs; **diversity** and **coverage** metrics.
- Online (pilot): A/B test for inquiry rate, adoption rate, and length-of-stay.

**Tiny demo sketch (pseudo-Python)**
```python
# 1) Build adopter embedding from profile text
adopter_text = to_text(profile_dict)
a_emb = sbert.encode([adopter_text])

# 2) Pet embeddings (precomputed)
P = np.vstack([pets[i].text_emb*0.6 + pets[i].image_emb*0.4 for i in range(len(pets))])

# 3) Candidate retrieval by cosine similarity
sims = (P @ a_emb.T).ravel() / (np.linalg.norm(P,axis=1)*np.linalg.norm(a_emb))
cand_idx = sims.argsort()[-100:][::-1]

# 4) Re-rank with features
X = build_rank_features(adopter=profile, pets=[pets[i] for i in cand_idx])
scores = ltr_model.predict(X)
rank = [cand_idx[i] for i in np.argsort(-scores)]
```

## Challenges

- Bias & fairness: avoid amplifying breed stereotypes or text biases; ensure equal exposure across looks/colors; monitor fairness metrics and provide override tools to staff.
- Data quality: inconsistent descriptions, missing tags, variable photo quality.
- Privacy: store minimal adopter data; consent for communications; GDPR/CCPA-aware.
- API limits: rate limiting and availability; cache and back-off strategies.
- Dynamic inventory: pets get adopted quickly; frequent refresh required.

## What next?

- MVP with public sandbox API keys (Petfinder) and demo embeddings.
- Add explanations (“Matched because low-energy + apartment-friendly”).
- Diversity objective (MMR) to avoid near-duplicate suggestions.
- Staff tools: priority queue for long-stay or special-needs pets.
- Partnerships with 1–2 local shelters for a supervised pilot and outcome tracking.


## Data sources and AI methods

| Source        | Fields                                   | Notes                 |
|---------------|-------------------------------------------|-----------------------|
| Petfinder API | species, size, age, tags, photos, status | caching; respect TOS  |
| Adopter form  | needs/preferences, constraints            | minimal PII, consent  |
| Embeddings    | SBERT/CLIP                                | HF models; CPU-friendly |


## Acknowledgments

*   Petfinder API and participating shelters.
    
*   Open-source tools: Hugging Face Transformers, Sentence-BERT, CLIP, scikit-learn/XGBoost, FastAPI.
    
*   Inspiration from humane tech and shelter communities committed to reducing length-of-stay.
