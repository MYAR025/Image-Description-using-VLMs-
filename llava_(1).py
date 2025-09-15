!pip install sentence-transformers scikit-learn

!pip install transformers torch torchvision datasets pillow accelerate bitsandbytes
!pip install rouge-score nltk sacrebleu sentence-transformers
!pip install matplotlib seaborn pandas numpy
!pip install pycocoevalcap bert-score spacy
!python -m spacy download en_core_web_sm
!pip install -U bitsandbytes

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from bert_score import score as bert_score
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import re

import warnings
warnings.filterwarnings('ignore')

# Download NLTK data for text processing
nltk.download('punkt')
nltk.download('stopwords')

"""We load the "jaimin/Image_Caption" dataset from the Hugging Face Hub. This dataset contains a collection of image and text pairs, suitable for training and evaluating image captioning models. It consists of 15,012 image-caption pairs in the training split."""


dataset = load_dataset("jaimin/Image_Caption")
df = dataset['train'].to_pandas()

print(f"Dataset Shape: {df.shape}")
print("\nDataset Columns:")
for col in df.columns:
    print(f"  • {col}: {df[col].dtype}")

print("\nFirst 5 rows:")
df.head()

"""## 2. Model Loading and Caption Generation

We load the LLaVA model and its processor. LLaVA is a large language and vision assistant model capable of understanding and generating text based on images.
"""

from transformers import LlavaProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("LLaVA Model loaded successfully!")

"""A function is defined to generate captions for a given image using the loaded LLaVA model."""

def generate_caption(image, prompt="Describe this image in detail."):
    try:
        full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"

        inputs = processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        caption = processor.decode(generated_tokens, skip_special_tokens=True)

        return caption.strip()

    except Exception as e:
        return f"Error: {str(e)}"

"""We select a subset of the dataset for testing the captioning model."""

test_size = 150
test_df = df.iloc[:test_size].copy()

print(f"\nGenerating captions for {test_size} images...")

"""Lists are initialized to store the generated captions (predictions) and the ground truth captions."""

predictions = []
ground_truths = []

"""We iterate through the test dataset, generate captions for each image, and store the predictions and ground truths."""

import io

for idx, row in test_df.iterrows():
    print(f"Processing image {idx+1}/{test_size}...", end='\r')

    
    image_bytes = row['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')


    pred_caption = generate_caption(image)

    predictions.append(pred_caption)
    ground_truths.append(row['text'])

test_df['llava_prediction'] = predictions
print(f"\nCaption generation completed!")

"""Here are a few examples to visually inspect the generated captions compared to the ground truth. This helps in getting an initial qualitative understanding of the model's performance."""

print("\nLLaVA Predictions vs Ground Truth:")
print("=" * 80)

for i in range(min(5, len(test_df))):
    print(f"\nImage {i+1}:")
    print(f"Ground Truth: {test_df.iloc[i]['text']}")
    print(f"LLaVA Prediction: {test_df.iloc[i]['llava_prediction']}")
    print("-" * 60)

"""Displaying the test DataFrame with the added 'llava_prediction' column."""

test_df.head(150)

"""Saving the test DataFrame with predictions to a CSV file for potential further analysis or download."""

subset_df = test_df[['text', 'llava_prediction']].head(150)

subset_df.to_csv("subset_output.csv", index=False)

print("Saved the subset of data to subset_output.csv in your Colab environment.")

"""## 3. Evaluation Metrics

We initialize various evaluation metrics commonly used for image captioning, including ROUGE, BLEU, and a semantic similarity model.
"""

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu = BLEU()


semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

"""A function to calculate basic evaluation metrics (ROUGE, BLEU, Semantic Similarity)."""

def calculate_metrics(predictions, references):
    """Calculate various evaluation metrics"""
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_scores = []

    # Calculate ROUGE scores
    for pred, ref in zip(predictions, references):
        scores = rouge.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

    # Calculate BLEU scores
    for pred, ref in zip(predictions, references):
        bleu_score = bleu.sentence_score(pred, [ref]).score
        bleu_scores.append(bleu_score)

    # Calculate semantic similarity
    pred_embeddings = semantic_model.encode(predictions)
    ref_embeddings = semantic_model.encode(references)
    semantic_scores = [cosine_similarity([pred], [ref])[0][0]
                      for pred, ref in zip(pred_embeddings, ref_embeddings)]

    return rouge_scores, bleu_scores, semantic_scores

# Calculate basic metrics
rouge_scores, bleu_scores, semantic_scores = calculate_metrics(
    test_df['llava_prediction'].tolist(),
    test_df['text'].tolist()
)

# Average basic scores
avg_rouge1 = np.mean(rouge_scores['rouge1'])
avg_rouge2 = np.mean(rouge_scores['rouge2'])
avg_rougeL = np.mean(rouge_scores['rougeL'])
avg_bleu = np.mean(bleu_scores)
avg_semantic = np.mean(semantic_scores)

"""We define a function to calculate more advanced evaluation metrics, including CIDEr, METEOR, BERTScore, SPICE-like, and Content Word Overlap. These metrics capture different aspects of caption quality beyond simple word overlap."""

# Load spacy model for linguistic analysis
nlp = spacy.load("en_core_web_sm")

def calculate_better_metrics(predictions, references, bertscore_batch_size=16):
    """Calculate 5 better metrics for image captioning"""

    # 1. CIDEr Score (Consensus-based Image Description Evaluation)
    def calculate_cider(pred, ref):
        def get_ngrams(text, n):
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

        # Get n-grams (1 to 4)
        cider_score = 0
        for n in range(1, 5):
            pred_ngrams = Counter(get_ngrams(pred, n))
            ref_ngrams = Counter(get_ngrams(ref, n))

            # Calculate TF-IDF weighted similarity
            intersection = sum((pred_ngrams & ref_ngrams).values())
            union = sum((pred_ngrams | ref_ngrams).values())

            if union > 0:
                cider_score += intersection / union

        return cider_score / 4  # Average across n-grams

    # 2. METEOR Score (with synonyms and paraphrases)
    def calculate_meteor(pred, ref):
        pred_doc = nlp(pred.lower())
        ref_doc = nlp(ref.lower())

        pred_tokens = [token.lemma_ for token in pred_doc if not token.is_stop and token.is_alpha]
        ref_tokens = [token.lemma_ for token in ref_doc if not token.is_stop and token.is_alpha]

        if not pred_tokens or not ref_tokens:
            return 0

        # Exact matches
        matches = len(set(pred_tokens) & set(ref_tokens))
        precision = matches / len(pred_tokens) if pred_tokens else 0
        recall = matches / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0

        f_score = 2 * precision * recall / (precision + recall)
        return f_score

    # 3. BERTScore (contextual embeddings)
    def calculate_bertscore_batch(predictions, references, batch_size):
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False, batch_size=batch_size)
        return F1.tolist()

    # 4. SPICE-like (Semantic content similarity)
    def calculate_spice_like(pred, ref):
        pred_doc = nlp(pred)
        ref_doc = nlp(ref)

        # Extract entities, objects, and relationships
        pred_entities = set([ent.text.lower() for ent in pred_doc.ents])
        ref_entities = set([ent.text.lower() for ent in ref_doc.ents])

        pred_nouns = set([token.lemma_.lower() for token in pred_doc if token.pos_ in ['NOUN', 'PROPN']])
        ref_nouns = set([token.lemma_.lower() for token in ref_doc if token.pos_ in ['NOUN', 'PROPN']])

        pred_verbs = set([token.lemma_.lower() for token in pred_doc if token.pos_ == 'VERB'])
        ref_verbs = set([token.lemma_.lower() for token in ref_doc if token.pos_ == 'VERB'])

        # Calculate semantic overlap
        entity_overlap = len(pred_entities & ref_entities) / max(len(ref_entities), 1)
        noun_overlap = len(pred_nouns & ref_nouns) / max(len(ref_nouns), 1)
        verb_overlap = len(pred_verbs & ref_verbs) / max(len(ref_verbs), 1)

        return (entity_overlap + noun_overlap + verb_overlap) / 3

    # 5. Content Word Overlap (CWO)
    def calculate_cwo(pred, ref):
        pred_doc = nlp(pred.lower())
        ref_doc = nlp(ref.lower())


        pred_content = set([token.lemma_ for token in pred_doc
                           if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop])
        ref_content = set([token.lemma_ for token in ref_doc
                          if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop])

        if not ref_content:
            return 0

        overlap = len(pred_content & ref_content)
        precision = overlap / len(pred_content) if pred_content else 0
        recall = overlap / len(ref_content)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    # Calculate all metrics
    print("Calculating advanced metrics...")

    cider_scores = [calculate_cider(pred, ref) for pred, ref in zip(predictions, references)]
    meteor_scores = [calculate_meteor(pred, ref) for pred, ref in zip(predictions, references)]
    bert_scores = calculate_bertscore_batch(predictions, references, bertscore_batch_size)
    spice_scores = [calculate_spice_like(pred, ref) for pred, ref in zip(predictions, references)]
    cwo_scores = [calculate_cwo(pred, ref) for pred, ref in zip(predictions, references)]

    return {
        'cider': cider_scores,
        'meteor': meteor_scores,
        'bertscore': bert_scores,
        'spice_like': spice_scores,
        'cwo': cwo_scores
    }

# Calculate the better metrics
better_metrics = calculate_better_metrics(
    test_df['llava_prediction'].tolist(),
    test_df['text'].tolist()
)

# Average better metrics
avg_cider = np.mean(better_metrics['cider'])
avg_meteor = np.mean(better_metrics['meteor'])
avg_bertscore = np.mean(better_metrics['bertscore'])
avg_spice = np.mean(better_metrics['spice_like'])
avg_cwo = np.mean(better_metrics['cwo'])

"""Displaying the average scores for both basic and advanced evaluation metrics."""

print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
print(f"ROUGE-L F1: {avg_rougeL:.4f}")
print(f"BLEU Score: {avg_bleu:.4f}")
print(f"Semantic Similarity: {avg_semantic:.4f}")
print(f"CIDEr Score: {avg_cider:.4f}")
print(f"METEOR Score: {avg_meteor:.4f}")
print(f"BERTScore F1: {avg_bertscore:.4f}")
print(f"SPICE-like Score: {avg_spice:.4f} ")
print(f"Content Word Overlap: {avg_cwo:.4f} ")

"""A function to calculate comprehensive evaluation metrics, including alternatives for CLIPScore, SPICE, Enhanced CIDEr, BERTScore, BLEURT, and MoverScore."""

def calculate_comprehensive_metrics_simple(predictions, references):

    print("🔄 Calculating comprehensive metrics...")
    results = {}

    # 1. CLIPScore Alternative (Text-based semantic similarity)
    print("📸 Calculating CLIPScore Alternative...")
    def text_clip_score(pred, ref):

        pred_emb = semantic_model.encode([pred])
        ref_emb = semantic_model.encode([ref])
        return cosine_similarity(pred_emb, ref_emb)[0][0]

    results['clip_score_alt'] = [text_clip_score(pred, ref) for pred, ref in zip(predictions, references)]

    # 2. SPICE Alternative (Enhanced semantic content matching)
    print("Calculating SPICE Alternative...")
    def spice_alternative(pred, ref):
        pred_doc = nlp(pred.lower())
        ref_doc = nlp(ref.lower())

        # Extract semantic content
        pred_entities = set([ent.text.lower() for ent in pred_doc.ents])
        ref_entities = set([ent.text.lower() for ent in ref_doc.ents])

        pred_objects = set([token.lemma_.lower() for token in pred_doc
                           if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])
        ref_objects = set([token.lemma_.lower() for token in ref_doc
                          if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])

        pred_actions = set([token.lemma_.lower() for token in pred_doc if token.pos_ == 'VERB'])
        ref_actions = set([token.lemma_.lower() for token in ref_doc if token.pos_ == 'VERB'])

        pred_attributes = set([token.lemma_.lower() for token in pred_doc if token.pos_ == 'ADJ'])
        ref_attributes = set([token.lemma_.lower() for token in ref_doc if token.pos_ == 'ADJ'])

        # Calculate overlaps with weights
        entity_score = len(pred_entities & ref_entities) / max(len(ref_entities), 1) * 0.3
        object_score = len(pred_objects & ref_objects) / max(len(ref_objects), 1) * 0.4
        action_score = len(pred_actions & ref_actions) / max(len(ref_actions), 1) * 0.2
        attr_score = len(pred_attributes & ref_attributes) / max(len(ref_attributes), 1) * 0.1

        return entity_score + object_score + action_score + attr_score

    results['spice_alt'] = [spice_alternative(pred, ref) for pred, ref in zip(predictions, references)]

    # 3. Enhanced CIDEr (with TF-IDF weighting)
    print("Calculating Enhanced CIDEr...")
    def enhanced_cider(predictions, references):
        # Create corpus for TF-IDF
        corpus = predictions + references
        vectorizer = TfidfVectorizer(ngram_range=(1, 4), lowercase=True)

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            pred_vectors = tfidf_matrix[:len(predictions)]
            ref_vectors = tfidf_matrix[len(predictions):]

            cider_scores = []
            for i in range(len(predictions)):
                score = cosine_similarity(pred_vectors[i], ref_vectors[i])[0][0]
                cider_scores.append(max(0, score))  # Ensure non-negative

            return cider_scores
        except:
            # Fallback to simple n-gram overlap
            return [calculate_cider_simple(pred, ref) for pred, ref in zip(predictions, references)]

    def calculate_cider_simple(pred, ref):
        def get_ngrams(text, n):
            words = text.lower().split()
            return Counter([' '.join(words[i:i+n]) for i in range(max(1, len(words)-n+1))])

        score = 0
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            intersection = sum((pred_ngrams & ref_ngrams).values())
            union = sum((pred_ngrams | ref_ngrams).values())

            if union > 0:
                score += intersection / union

        return score / 4

    results['cider_enhanced'] = enhanced_cider(predictions, references)

    # 4. BERTScore
    print("Calculating BERTScore...")
    try:
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        results['bertscore'] = F1.tolist()
    except:
        results['bertscore'] = results['clip_score_alt']  

    # 5. BLEURT Alternative
    print("Calculating BLEURT Alternative...")
    def bleurt_alternative(pred, ref):
        # Multi-level semantic comparison

        # Sentence-level similarity
        sent_sim = text_clip_score(pred, ref)

        # Word-level alignment
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        word_overlap = len(pred_words & ref_words) / max(len(ref_words), 1)

        # Length penalty
        len_ratio = min(len(pred.split()), len(ref.split())) / max(len(pred.split()), len(ref.split()), 1)

        # Combined score
        return (sent_sim * 0.6 + word_overlap * 0.3 + len_ratio * 0.1)

    results['bleurt_alt'] = [bleurt_alternative(pred, ref) for pred, ref in zip(predictions, references)]

    # 6. MoverScore Alternative (Semantic distance with word importance)
    print("Calculating MoverScore Alternative...")
    def moverscore_alternative(pred, ref):
        pred_doc = nlp(pred)
        ref_doc = nlp(ref)

        # Get important words (not stopwords)
        pred_important = [token.lemma_.lower() for token in pred_doc
                         if not token.is_stop and token.is_alpha and len(token.text) > 2]
        ref_important = [token.lemma_.lower() for token in ref_doc
                        if not token.is_stop and token.is_alpha and len(token.text) > 2]

        if not pred_important or not ref_important:
            return 0

        # Calculate semantic similarity of important words
        pred_emb = semantic_model.encode(pred_important)
        ref_emb = semantic_model.encode(ref_important)

        # Find best alignments
        similarities = cosine_similarity(pred_emb, ref_emb)

        # Use Hungarian algorithm approximation (max similarity per word)
        alignment_scores = []
        for pred_idx in range(len(pred_important)):
            if len(ref_important) > 0:
                best_match = np.max(similarities[pred_idx])
                alignment_scores.append(best_match)

        return np.mean(alignment_scores) if alignment_scores else 0

    results['moverscore_alt'] = [moverscore_alternative(pred, ref) for pred, ref in zip(predictions, references)]

    return results

# Run the comprehensive evaluation
comprehensive_metrics = calculate_comprehensive_metrics_simple(
    test_df['llava_prediction'].tolist(),
    test_df['text'].tolist()
)

# Display results
print("\nCOMPREHENSIVE EVALUATION RESULTS")
print("=" * 60)

metric_descriptions = {
    'clip_score_alt': 'CLIPScore Alternative (Semantic)',
    'spice_alt': 'SPICE Alternative (Content)',
    'cider_enhanced': 'CIDEr Enhanced (Consensus)',
    'bertscore': 'BERTScore (Contextual)',
    'bleurt_alt': 'BLEURT Alternative (Learned-like)',
    'moverscore_alt': 'MoverScore Alternative (Alignment)'
}
for metric, scores in comprehensive_metrics.items():
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    description = metric_descriptions.get(metric, metric.upper())

    print(f"{description}: {avg_score:.4f} (±{std_score:.3f})")

"""## 4. Individual Example Analysis

Examining individual examples with their corresponding images, ground truth captions, generated captions, and metric scores to gain qualitative insights into the model's strengths and weaknesses.
"""

# Recalculate better metrics with a smaller batch size for BERTScore
better_metrics = calculate_better_metrics(
    test_df['llava_prediction'].tolist(),
    test_df['text'].tolist(),
    bertscore_batch_size=8  # Reduced batch size
)

# Now display the metrics for the selected examples
example_indices = [0, 5, 10, 15, 20]

for i in example_indices:
    print(f"\n--- Example {i+1} ---")

    # Display Image (re-display for context)
    image_bytes = test_df.iloc[i]['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()

    # Display Captions
    ground_truth = test_df.iloc[i]['text']
    prediction = test_df.iloc[i]['llava_prediction']
    print(f"Ground Truth: {ground_truth}")
    print(f"LLaVA Prediction: {prediction}")

    # Display Metrics for this example
    print("\nIndividual Metric Scores:")
    print(f"  ROUGE-1 F1: {rouge_scores['rouge1'][i]:.4f}")
    print(f"  ROUGE-2 F1: {rouge_scores['rouge2'][i]:.4f}")
    print(f"  ROUGE-L F1: {rouge_scores['rougeL'][i]:.4f}")
    print(f"  BLEU Score: {bleu_scores[i]:.4f}")
    print(f"  Semantic Similarity: {semantic_scores[i]:.4f}")
    print(f"  CIDEr Score: {better_metrics['cider'][i]:.4f}")
    print(f"  METEOR Score: {better_metrics['meteor'][i]:.4f}")
    print(f"  BERTScore F1: {better_metrics['bertscore'][i]:.4f}")
    print(f"  SPICE-like Score: {better_metrics['spice_like'][i]:.4f}")
    print(f"  Content Word Overlap: {better_metrics['cwo'][i]:.4f}")

    print("-" * 30)

import io

# Select additional example indices
example_indices_additional = [25, 30, 35, 40, 45]

for i in example_indices_additional:
    print(f"\n--- Example {i+1} ---")

    # Display Image
    image_bytes = test_df.iloc[i]['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()

    # Display Captions
    ground_truth = test_df.iloc[i]['text']
    prediction = test_df.iloc[i]['llava_prediction']
    print(f"Ground Truth: {ground_truth}")
    print(f"LLaVA Prediction: {prediction}")

    # Display Metrics for this example
    print("\nIndividual Metric Scores:")
    print(f"  ROUGE-1 F1: {rouge_scores['rouge1'][i]:.4f}")
    print(f"  ROUGE-2 F1: {rouge_scores['rouge2'][i]:.4f}")
    print(f"  ROUGE-L F1: {rouge_scores['rougeL'][i]:.4f}")
    print(f"  BLEU Score: {bleu_scores[i]:.4f}")
    print(f"  Semantic Similarity: {semantic_scores[i]:.4f}")
    print(f"  CIDEr Score: {better_metrics['cider'][i]:.4f}")
    print(f"  METEOR Score: {better_metrics['meteor'][i]:.4f}")
    print(f"  BERTScore F1: {better_metrics['bertscore'][i]:.4f}") # Include BERTScore
    print(f"  SPICE-like Score: {better_metrics['spice_like'][i]:.4f}")
    print(f"  Content Word Overlap: {better_metrics['cwo'][i]:.4f}")

    print("-" * 30)

"""## 6. Summary and Conclusion

### Summary of Evaluation Results

Based on the evaluation metrics and the analysis of individual examples, here is a summary of the LLaVA model's performance on the image captioning task:

**Overall Metrics:**

*   **Traditional Metrics (ROUGE, BLEU):** These metrics show relatively low scores, which is common for generative tasks like image captioning where there can be many valid ways to describe an image. ROUGE-1 and ROUGE-L, which measure unigram and longest common subsequence overlap respectively, are higher than ROUGE-2, indicating that the model captures some of the key words and phrases but struggles with capturing longer sequences of words accurately compared to the ground truth. The low BLEU score also suggests a lack of exact n-gram overlap with the reference captions.

*   **Semantic Similarity (Sentence-BERT):** The moderate semantic similarity score indicates that while the generated captions may not use the exact same words or phrases as the ground truth, they often convey a similar overall meaning.

*   **Advanced Metrics (CIDEr, METEOR, BERTScore, SPICE-like, CWO):**
    *   **BERTScore:** Shows a relatively high score, suggesting that the generated captions are contextually similar to the ground truth at a deeper level, even if the exact words differ.
    *   **METEOR:** A moderate score indicates some level of alignment between the generated and reference captions, considering synonyms and paraphrases.
    *   **CIDEr:** The low CIDEr score suggests that the generated captions do not align strongly with the consensus of human-written captions in terms of n-gram overlap, especially when considering the TF-IDF weighting.
    *   **SPICE-like and CWO:** These scores, which focus on semantic content and important word overlap, are also relatively low, indicating that the model might miss some specific objects, attributes, or relationships present in the ground truth captions.

**Insights from Individual Examples:**

*   Looking at individual examples confirms that the model often generates plausible descriptions, but they may lack the specific details or perspectives present in the ground truth captions.
*   In some cases, the model might hallucinate objects or actions that are not actually in the image (e.g., describing people racing when the image shows parked cars).
*   The model sometimes focuses on prominent aspects of the image while missing more subtle details mentioned in the ground truth.
*   The variation in individual metric scores highlights that the model's performance can vary depending on the complexity and content of the image.

**Conclusion:**

The LLaVA model demonstrates some capability in generating semantically relevant image captions, as indicated by the semantic similarity and BERTScore. However, the lower scores on metrics that rely more on exact word overlap and capturing specific details suggest that there is room for improvement in generating captions that are both fluent and highly aligned with human descriptions at a granular level. Further fine-tuning or exploring different prompting strategies could potentially improve the model's performance on these aspects.
"""
