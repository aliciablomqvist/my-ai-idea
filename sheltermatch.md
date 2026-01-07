ShelterMatch
============

**Final project for the Building AI course**

Summary
-------

ShelterMatch is a **student project** that explores how AI can be used to recommend adoptable pets to people based on their preferences and living situation.

The system matches adopter profiles with pet profiles (text and images) using embeddings and a simple learning-to-rank approach. The goal is to suggest pets that are a good fit for a household, which could help shelters reduce how long animals stay in care and improve the chances of successful adoptions.

Background
----------

Millions of animals enter shelters every year. Many adopters browse listings randomly, while shelter staff often make matches based on experience and limited time.

This process can be slow and may miss good matches, for example:

*   quiet or shy pets that would fit well in calm homes
    
*   animals that are overlooked because of appearance or limited descriptions
    

An AI-assisted recommendation system could help by:

*   improving visibility for less popular animals
    
*   reducing time-to-adoption
    
*   supporting better matches and fewer returns
    

This project investigates how such a system **could** work in practice.

How is it used?
---------------

**Adopter profile (5–10 short questions):**

*   home type (apartment/house), yard
    
*   children or other pets
    
*   activity level
    
*   grooming tolerance and allergies
    
*   preferred size and age
    
*   training expectations
    
*   location and distance radius
    

**Pet data (from shelter APIs such as Petfinder):**

*   species, breed mix, size, age, sex
    
*   compatibility information
    
*   medical or behavioral notes
    
*   photos
    

**Ranking process:**

*   similarity is calculated between adopter preferences and pet profiles using text and image features
    
*   a learning-to-rank model adjusts results based on factors such as distance and compatibility
    

**User interface:**

*   a web page where adopters can browse recommendations
    
*   explanations such as “Good match for quiet homes” or “Low shedding”
    
*   a simple contact or visit flow
    

**Staff view (conceptual):**

*   highlights pets with strong matches
    
*   helps staff prioritize outreach
    

**Feedback loop:**

*   user actions such as saves, likes, and inquiries are used to improve future recommendations
    

**Users:** adopters and shelter staff**Context:** mobile-friendly web application

Data sources and AI methods
---------------------------

### Data

*   **Pet data:** Petfinder API (profiles, attributes, images, location, adoption status)
    
*   **Adopter data:** voluntary questionnaire with minimal personal data (email only for follow-up)
    
*   **Additional data:** city/ZIP geocoding and basic shelter information
    

### Features

*   **Text:** pet descriptions and tags converted into sentence embeddings (e.g. SBERT)
    
*   **Images:** pet photos converted into image embeddings (e.g. CLIP) with simple quality checks
    
*   **Structured data:** species, age, size, distance, compatibility flags, energy level
    
*   **Adopter features:** activity level, home constraints, allergies, size and age preferences
    

### Models

**Stage 1 – Candidate retrieval:**

*   cosine similarity between adopter embeddings and pet embeddings (text and image features combined)
    
*   returns a shortlist of top candidates (e.g. top 100 pets)
    

**Stage 2 – Re-ranking:**

*   learning-to-rank model (e.g. LambdaMART or XGBoostRanker)
    
*   uses features such as distance, compatibility, historical adoption likelihood, and photo quality
    

**Cold start:**

*   rule-based defaults and popularity-based suggestions
    
*   personalization improves as users interact with results
    

### Evaluation

*   **Offline evaluation:** NDCG@k, MRR, Precision@k using historical inquiry or adoption data
    
*   **Online evaluation (conceptual):** A/B testing to compare inquiry rate, adoption rate, and length-of-stay
    
## Tiny demo sketch (pseudo-Python)

```python
# 1) Build adopter embedding from profile text
adopter_text = to_text(profile_dict)
a_emb = sbert.encode([adopter_text])

# 2) Pet embeddings (precomputed)
P = np.vstack([
    pets[i].text_emb * 0.6 + pets[i].image_emb * 0.4
    for i in range(len(pets))
])

# 3) Candidate retrieval by cosine similarity
sims = (P @ a_emb.T).ravel() / (
    np.linalg.norm(P, axis=1) * np.linalg.norm(a_emb)
)
cand_idx = sims.argsort()[-100:][::-1]

# 4) Re-rank with additional features
X = build_rank_features(
    adopter=profile,
    pets=[pets[i] for i in cand_idx]
)
scores = ltr_model.predict(X)
rank = [cand_idx[i] for i in np.argsort(-scores)]
```

Challenges
----------

*   **Bias and fairness:** risk of reinforcing stereotypes; need to ensure fair exposure across different animals
    
*   **Data quality:** missing tags, inconsistent descriptions, and uneven photo quality
    
*   **Privacy:** minimal data storage, consent-based communication, GDPR/CCPA awareness
    
*   **API limitations:** rate limits and availability; need for caching
    
*   **Dynamic inventory:** pets may be adopted quickly and require frequent updates
    

What next?
----------

*   Build an MVP using public sandbox API keys
    
*   Add clearer explanations for recommendations
    
*   Improve diversity in suggestions to avoid near-duplicates
    
*   Add simple tools for shelter staff
    
*   Explore a small supervised pilot with one or two local shelters
    

## Data sources and AI methods

| Source         | Fields                                   | Notes                     |
|---------------|------------------------------------------|---------------------------|
| Petfinder API | species, size, age, tags, photos, status | caching, respect TOS      |
| Adopter form  | needs, preferences, constraints          | minimal personal data     |
| Embeddings    | SBERT, CLIP                              | CPU-friendly models       |

Acknowledgments
---------------

*   Petfinder API and participating shelters
    
*   Open-source tools: Hugging Face Transformers, Sentence-BERT, CLIP, scikit-learn, XGBoost, FastAPI
    
*   Inspiration from humane tech and animal welfare communities focused on reducing shelter length-of-stay
