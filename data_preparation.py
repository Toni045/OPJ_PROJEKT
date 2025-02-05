# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import sent_tokenize
import torch
from torch.utils.data import Dataset
nltk.download('punkt')

class DataPreparator:
    def __init__(self):
        pass

    def custom_sentence_split(self, text):
        """Custom sentence splitting using NLTK"""
        return sent_tokenize(text)  # This uses NLTK's sentence tokenizer

    def prepare_tourism_data(self, file_path):
        """Prepare tourism data from CSV and return train, validation, and test sets"""
        # Load the CSV data
        data = pd.read_csv(file_path)
        
        # Create question-answer pairs from the text column (assuming 'text' is the column with tourism information)
        qa_pairs = []
        for _, row in data.iterrows():
            text = row['text']  # Assuming the text data is in the 'text' column
            qa_pairs.extend(self.create_qa_pairs(text))
        
        # Convert QA pairs to a DataFrame
        qa_df = pd.DataFrame(qa_pairs)
        
        # Split data into train, validation, and test sets (80% train, 10% validation, 10% test)
        train_data, temp_data = train_test_split(qa_df, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        return train_data, val_data, test_data
    
    def create_qa_pairs(self, text):
        """Poboljšano kreiranje parova pitanje-odgovor"""
        sentences = self.custom_sentence_split(text)
        qa_pairs = []
        
        for i, sentence in enumerate(sentences):
            # Kreiranje konteksta od okolnih rečenica
            start_idx = max(0, i - 1)
            end_idx = min(len(sentences), i + 2)
            context = ' '.join(sentences[start_idx:end_idx])
            
            # Poboljšani obrasci za generiranje pitanja
            if len(sentence.split()) > 5:
                patterns = [
                    # Obrazac za lokaciju
                    (r'([^,\.]+) (?:se nalazi|je smješten[a]?) ([^,\.]+)',
                     lambda m: f"Gdje se nalazi {m.group(1)}?",
                     lambda m: f"{m.group(1)} {m.group(2)}"),
                    
                    # Obrazac za opise
                    (r'([^,\.]+) je ([^,\.]+)',
                     lambda m: f"Što je {m.group(1)}?",
                     lambda m: f"{m.group(1)} je {m.group(2)}"),
                    
                    # Obrazac za znamenitosti
                    (r'([^,\.]+) (?:ima|sadrži|uključuje) ([^,\.]+)',
                     lambda m: f"Što {m.group(1)} ima?",
                     lambda m: f"{m.group(1)} ima {m.group(2)}")
                ]
                
                for pattern, q_gen, a_gen in patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        question = q_gen(match)
                        answer = a_gen(match)
                        
                        # Provjera kvalitete pitanja i odgovora
                        if (len(question.split()) >= 3 and 
                            len(answer.split()) >= 3 and 
                            answer not in qa_pairs):
                            
                            qa_pairs.append({
                                'question': question,
                                'answer': answer,
                                'context': context
                            })
        
        return qa_pairs

# models.py

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Bolje označavanje odgovora
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Pronalaženje točnih pozicija odgovora u tokeniziranom tekstu
        answer_start = item['context'].lower().find(item['answer'].lower())
        answer_end = answer_start + len(item['answer'])
        
        # Pretvaranje character pozicija u token pozicije
        token_start = None
        token_end = None
        offset_mapping = encoding.offset_mapping[0].tolist()
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                token_start = idx
            if start <= answer_end <= end and token_start is not None:
                token_end = idx
                break
                
        # Ako ne možemo naći točne pozicije, koristimo prve tokene
        if token_start is None or token_end is None:
            token_start = 1  # Prvi token nakon [CLS]
            token_end = 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(token_start, dtype=torch.long),
            'end_positions': torch.tensor(token_end, dtype=torch.long)
        }

# evaluation.py

class Evaluator:
    def find_best_answer(self, start_logits, end_logits, input_ids, tokenizer, max_answer_length=50):
        """Poboljšano pronalaženje najboljeg odgovora"""
        # Pretvaramo u numpy za lakše rukovanje
        start_logits = start_logits[0].cpu().numpy()
        end_logits = end_logits[0].cpu().numpy()
        
        # Pronalazimo najbolje početne i završne pozicije
        start_idx = np.argsort(start_logits)[-20:][::-1]  # Top 20 početaka
        end_idx = np.argsort(end_logits)[-20:][::-1]  # Top 20 krajeva
        
        best_score = float('-inf')
        best_answer = ""
        
        # Isprobavamo sve valjane kombinacije
        for start in start_idx:
            for end in end_idx:
                if start > end or end - start + 1 > max_answer_length:
                    continue
                    
                # Provjera da odgovor ne počinje ili završava s special tokenima
                if start == 0 or end == len(input_ids[0]) - 1:
                    continue
                    
                score = start_logits[start] + end_logits[end]
                if score > best_score:
                    tokens = input_ids[0][start:end + 1]
                    answer = tokenizer.decode(tokens, skip_special_tokens=True)
                    
                    # Provjera kvalitete odgovora
                    if (len(answer.split()) >= 2 and  # Minimalno 2 riječi
                        not answer.startswith('?') and  # Ne počinje s upitnikom
                        not any(q in answer.lower() for q in ['što', 'gdje', 'kada', 'tko'])):  # Ne sadrži upitne riječi
                        
                        best_score = score
                        best_answer = answer
        
        return best_answer.strip()

    def compute_metrics(self, results, prediction, reference):
        """Poboljšano računanje metrika"""
        try:
            # Exact match
            results['exact_match'].append(prediction.lower() == reference.lower())
            
            # F1 score
            pred_tokens = set(prediction.lower().split())
            ref_tokens = set(reference.lower().split())
            
            if not pred_tokens or not ref_tokens:
                results['f1'].append(0.0)
            else:
                intersection = pred_tokens & ref_tokens
                precision = len(intersection) / len(pred_tokens)
                recall = len(intersection) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                results['f1'].append(f1)
            
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
            
            # Dodajemo nule umjesto nan vrijednosti
            results['exact_match'].append(0.0)
            results['f1'].append(0.0)
            results['rouge'].append(0.0)
            results['tourism_relevance'].append(0.0)
            results['factual_accuracy'].append(0.0)