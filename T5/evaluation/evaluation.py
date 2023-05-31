import os
import torch
from evaluation.metrics import evaluate_style_transfer, rotation_calibration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from nltk.translate.chrf_score import corpus_chrf
import pandas as pd


def load_model(model_name=None, model=None, tokenizer=None,
               model_class=AutoModelForSequenceClassification, use_cuda=True):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def evaluate(original, rewritten, neutral_references, batch_size):
    print('Evaluation... Models loading...')
    
    style_model, style_tokenizer = load_model(
        'SkolkovoInstitute/russian_toxicity_classifier', 
        use_cuda=True
    )
    meaning_model, meaning_tokenizer = load_model(
        'cointegrated/LaBSE-en-ru', 
        use_cuda=True,
        model_class=AutoModel
    )
    fluency_model, fluency_tolenizer = load_model(
        'SkolkovoInstitute/rubert-base-corruption-detector',
        use_cuda=True
    )
        
    results = evaluate_style_transfer(
        original_texts=original,
        rewritten_texts=rewritten,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=fluency_model,
        cola_tokenizer=fluency_tolenizer,
        style_target_label=0,
        batch_size=batch_size,
        aggregate=True
    )
    if neutral_references is not None:
        results['chrf'] = corpus_chrf(neutral_references, rewritten)
    
    return results