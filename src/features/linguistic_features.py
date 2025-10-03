"""
Linguistic Feature Extraction Module
- Extracts linguistic features from text for prosody prediction
- Handles POS tagging, dependency parsing, and linguistic analysis
- Supports emotion-aware feature extraction
"""

import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import spacy
import numpy as np

class LinguisticFeatureExtractor:
    def __init__(self, 
                 bert_model: str = "bert-base-uncased",
                 spacy_model: str = "en_core_web_sm"):
        # Initialize BERT for semantic features
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        
        # Initialize spaCy for linguistic analysis
        self.nlp = spacy.load(spacy_model)
        
        # POS tag groups for feature extraction
        self.pos_groups = {
            'content': ['NOUN', 'VERB', 'ADJ', 'ADV'],
            'function': ['ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON'],
            'structure': ['PUNCT', 'SCONJ', 'INTJ']
        }
        
    def extract_features(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive linguistic features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing:
            - bert_embeddings: Contextual word embeddings
            - pos_features: Part-of-speech features
            - dep_features: Dependency parsing features
            - prosodic_features: Features relevant for prosody
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Get BERT embeddings
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
        bert_embeddings = bert_outputs.last_hidden_state[0]  # Remove batch dim
        
        # Extract POS features
        pos_features = []
        for token in doc:
            pos_vector = [0] * len(self.pos_groups)
            for i, (group, tags) in enumerate(self.pos_groups.items()):
                if token.pos_ in tags:
                    pos_vector[i] = 1
            pos_features.append(pos_vector)
        pos_features = torch.tensor(pos_features)
        
        # Extract dependency features
        dep_features = []
        for token in doc:
            # One-hot encoding of dependency relation
            dep_vector = [0] * len(spacy.symbols.dep)
            dep_vector[token.dep] = 1
            dep_features.append(dep_vector)
        dep_features = torch.tensor(dep_features)
        
        # Extract prosodic features
        prosodic_features = self._extract_prosodic_features(doc)
        
        return {
            'bert_embeddings': bert_embeddings,
            'pos_features': pos_features,
            'dep_features': dep_features,
            'prosodic_features': prosodic_features
        }
    
    def _extract_prosodic_features(self, doc) -> torch.Tensor:
        """Extract features specifically relevant for prosody prediction"""
        features = []
        
        for token in doc:
            token_features = []
            
            # Word position features
            sent_progress = token.i / len(doc)
            token_features.append(sent_progress)
            
            # Punctuation context
            next_punct = any(t.pos_ == 'PUNCT' for t in token.rights)
            prev_punct = any(t.pos_ == 'PUNCT' for t in token.lefts)
            token_features.extend([float(next_punct), float(prev_punct)])
            
            # Syntactic importance
            is_root = token.dep_ == 'ROOT'
            depth = len(list(token.ancestors))
            n_children = len(list(token.children))
            token_features.extend([float(is_root), depth, n_children])
            
            # Word emphasis features
            is_content = token.pos_ in self.pos_groups['content']
            is_capitalized = token.text[0].isupper() if token.text else False
            token_features.extend([float(is_content), float(is_capitalized)])
            
            features.append(token_features)
            
        return torch.tensor(features)
    
    def get_emotion_linguistic_features(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Extract linguistic features specifically relevant for emotion
        
        Args:
            text: Input text
            
        Returns:
            Emotion-relevant linguistic features
        """
        doc = self.nlp(text)
        features = []
        
        for token in doc:
            token_features = []
            
            # Emotional word indicators
            is_intensifier = token.dep_ == 'advmod' and token.head.pos_ == 'ADJ'
            is_negation = token.dep_ == 'neg'
            is_exclamation = token.text in '!?'
            
            # Syntactic emotion indicators
            has_intensifier = any(child.dep_ == 'advmod' for child in token.children)
            is_emphasized = token.text.isupper()
            
            token_features.extend([
                float(is_intensifier),
                float(is_negation),
                float(is_exclamation),
                float(has_intensifier),
                float(is_emphasized)
            ])
            
            features.append(token_features)
            
        return {
            'emotion_features': torch.tensor(features),
            'sentence_features': self._get_sentence_emotion_features(doc)
        }
    
    def _get_sentence_emotion_features(self, doc) -> torch.Tensor:
        """Extract sentence-level emotion features"""
        features = []
        
        # Sentence type
        is_question = any(token.text == '?' for token in doc)
        is_exclamation = any(token.text == '!' for token in doc)
        
        # Sentence structure
        has_intensifiers = any(token.dep_ == 'advmod' for token in doc)
        has_negation = any(token.dep_ == 'neg' for token in doc)
        
        # Emotional content
        capitalized_ratio = sum(t.text[0].isupper() for t in doc) / len(doc)
        
        features.extend([
            float(is_question),
            float(is_exclamation),
            float(has_intensifiers),
            float(has_negation),
            capitalized_ratio
        ])
        
        return torch.tensor(features)