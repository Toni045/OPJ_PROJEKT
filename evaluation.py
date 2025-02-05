# evaluation.py
import evaluate
from tqdm import tqdm
import torch
import re
import numpy as np
from transformers import default_data_collator

class Evaluator:
    def __init__(self):
        self.metrics = {
            'exact_match': evaluate.load('exact_match'),
            'f1': evaluate.load('f1'),
            'rouge': evaluate.load('rouge')
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluator koristi uređaj: {self.device}")
    
    def find_best_answer(self, start_logits, end_logits, input_ids, tokenizer, max_answer_length=50):
        """Pronalazi najbolji odgovor iz logitsa"""
        # Pretvaramo u numpy za lakše rukovanje
        start_logits = start_logits[0].cpu().numpy()
        end_logits = end_logits[0].cpu().numpy()
        
        # Pronalazimo najbolje početne i završne pozicije
        start_idx = np.argsort(start_logits)[-10:][::-1]  # Top 10 početaka
        end_idx = np.argsort(end_logits)[-10:][::-1]  # Top 10 krajeva
        
        best_score = float('-inf')
        best_answer = ""
        
        # Isprobavamo sve kombinacije
        for start in start_idx:
            for end in end_idx:
                if end < start or end - start + 1 > max_answer_length:
                    continue
                    
                score = start_logits[start] + end_logits[end]
                if score > best_score:
                    best_score = score
                    tokens = input_ids[0][start:end + 1]
                    answer = tokenizer.decode(tokens, skip_special_tokens=True)
                    if answer.strip():  # Provjera da odgovor nije prazan
                        best_answer = answer
        
        return best_answer.strip()

    def evaluate_model(self, model, tokenizer, test_data):
        """Evaluacija modela na test skupu"""
        print("\nPočetak evaluacije modela...")
        model.eval()
        model.to(self.device)
        
        results = {
            'exact_match': [],
            'f1': [],
            'rouge': [],
            'tourism_relevance': [],
            'factual_accuracy': []
        }
        
        all_predictions = []
        all_references = []
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            try:
                # Tokenizacija
                inputs = tokenizer(
                    row['question'],
                    row['context'],
                    max_length=384,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Prebacivanje na GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generiranje predviđanja
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Pronalaženje najboljeg odgovora
                prediction = self.find_best_answer(
                    outputs.start_logits,
                    outputs.end_logits,
                    inputs['input_ids'],
                    tokenizer
                )
                
                # Čišćenje predviđanja
                prediction = self.clean_prediction(prediction)
                reference = row['answer']
                
                # Spremanje za ukupnu evaluaciju
                if prediction and reference:
                    all_predictions.append(prediction)
                    all_references.append(reference)
                    
                    # Računanje metrika za pojedinačni primjer
                    self.compute_metrics(results, prediction, reference)
                    
                    print(f"\nPitanje: {row['question']}")
                    print(f"Predviđeni odgovor: {prediction}")
                    print(f"Točan odgovor: {reference}")
                
            except Exception as e:
                print(f"Greška pri evaluaciji primjera: {str(e)}")
                continue
        
        # Računanje prosječnih vrijednosti
        final_results = {}
        
        # Osnovne metrike
        if all_predictions:
            final_results['exact_match'] = np.mean(results['exact_match']) if results['exact_match'] else 0
            final_results['f1'] = np.mean(results['f1']) if results['f1'] else 0
            final_results['rouge'] = np.mean(results['rouge']) if results['rouge'] else 0
        else:
            final_results['exact_match'] = 0
            final_results['f1'] = 0
            final_results['rouge'] = 0
        
        # Custom metrike
        final_results['tourism_relevance'] = np.mean(results['tourism_relevance']) if results['tourism_relevance'] else 0
        final_results['factual_accuracy'] = np.mean(results['factual_accuracy']) if results['factual_accuracy'] else 0
        
        print("\nRezultati evaluacije:")
        for metric, value in final_results.items():
            print(f"{metric}: {value:.4f}")
        
        return final_results
    
    def clean_prediction(self, pred):
        """Čišćenje predviđenog odgovora"""
        if not isinstance(pred, str):
            return ""
            
        # Uklanjanje special tokena i tagova
        pred = re.sub(r'<s>|</s>|\[CLS\]|\[SEP\]|\[PAD\]', '', pred)
        # Uklanjanje nepotrebnih razmaka
        pred = ' '.join(pred.split())
        # Uklanjanje ##
        pred = pred.replace('##', '')
        return pred.strip()
    
    def compute_metrics(self, results, prediction, reference):
        """Računanje metrika za jedan primjer"""
        try:
            # Exact match
            em_score = self.metrics['exact_match'].compute(
                predictions=[prediction],
                references=[reference]
            )
            results['exact_match'].append(em_score['exact_match'])
            
            # F1 score
            f1_score = self.metrics['f1'].compute(
                predictions=[prediction],
                references=[reference]
            )
            results['f1'].append(f1_score['f1'])
            
            # ROUGE score
            rouge_scores = self.metrics['rouge'].compute(
                predictions=[prediction],
                references=[reference]
            )
            results['rouge'].append(rouge_scores['rouge1'].mid.fmeasure)
            
            # Custom metrike
            results['tourism_relevance'].append(
                self.evaluate_tourism_relevance(prediction, reference)
            )
            results['factual_accuracy'].append(
                self.evaluate_factual_accuracy(prediction, reference)
            )
            
        except Exception as e:
            print(f"\nGreška pri računanju metrika:")
            print(f"Predviđanje: {prediction}")
            print(f"Referenca: {reference}")
            print(f"Error: {str(e)}")
    
    def evaluate_tourism_relevance(self, prediction, reference):
        """Evaluacija relevantnosti za turističku domenu"""
        tourism_keywords = [
            'hotel', 'restoran', 'muzej', 'plaža', 'znamenitost',
            'atrakcija', 'izlet', 'tura', 'smještaj', 'transport',
            'grad', 'otok', 'park', 'jezero', 'more', 'planina',
            'crkva', 'katedrala', 'palača', 'tvrđava', 'dvorac',
            'festival', 'kultura', 'povijest', 'arhitektura'
        ]
        
        pred_keywords = sum(1 for keyword in tourism_keywords 
                          if keyword in prediction.lower())
        ref_keywords = sum(1 for keyword in tourism_keywords 
                         if keyword in reference.lower())
        
        if ref_keywords == 0:
            return 1.0 if pred_keywords == 0 else 0.0
        return min(pred_keywords / ref_keywords, 1.0)
    
    def evaluate_factual_accuracy(self, prediction, reference):
        """Evaluacija činjenične točnosti"""
        try:
            # Brojevi
            pred_numbers = set(re.findall(r'\d+', prediction))
            ref_numbers = set(re.findall(r'\d+', reference))
            
            # Imena i nazivi
            pred_names = set(re.findall(r'[A-ZČĆĐŠŽ][a-zčćđšž]+', prediction))
            ref_names = set(re.findall(r'[A-ZČĆĐŠŽ][a-zčćđšž]+', reference))
            
            # Računanje točnosti
            number_match = len(pred_numbers & ref_numbers)
            name_match = len(pred_names & ref_names)
            
            number_accuracy = number_match / len(ref_numbers) if ref_numbers else 1.0
            name_accuracy = name_match / len(ref_names) if ref_names else 1.0
            
            return (number_accuracy + name_accuracy) / 2
            
        except Exception as e:
            print(f"Greška u računanju činjenične točnosti: {str(e)}")
            return 0.0