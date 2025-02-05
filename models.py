# models.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os
import gc

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Skraćujemo kontekst ako je predugačak
        if len(item['context']) > self.max_length:
            item['context'] = item['context'][:self.max_length]
        
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Pronalaženje pozicije odgovora u kontekstu
        answer_start = item['context'].lower().find(item['answer'].lower())
        answer_end = answer_start + len(item['answer'])
        
        # Ako je odgovor izvan konteksta, koristimo početak
        if answer_start == -1:
            answer_start = 0
            answer_end = 1
        
        # Pretvaramo u torch tensore
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(answer_start, dtype=torch.long),
            'end_positions': torch.tensor(answer_end, dtype=torch.long)
        }

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        # Provjera dostupnosti GPU-a
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Koristi se uređaj: {self.device}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def initialize_models(self):
        """Inicijalizacija različitih modela"""
        model_configs = {
            'mBERT': 'bert-base-multilingual-cased',
            'XLM-RoBERTa': 'xlm-roberta-base'
        }
        
        for name, path in model_configs.items():
            try:
                print(f"Učitavanje modela {name}...")
                self.tokenizers[name] = AutoTokenizer.from_pretrained(path)
                model = AutoModelForQuestionAnswering.from_pretrained(path)
                
                # Eksplicitno prebacujemo model na uređaj
                model = model.to(self.device)
                self.models[name] = model
                
                print(f"Uspješno učitan model: {name}")
            except Exception as e:
                print(f"Greška pri učitavanju modela {name}: {str(e)}")
    
    def train_model(self, model_name, train_data, val_data, output_dir):
        """Treniranje određenog modela"""
        if model_name not in self.models:
            print(f"Model {model_name} nije pronađen!")
            return None
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Oslobađamo memoriju prije treniranja
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Priprema podataka
        train_dataset = QADataset(train_data, tokenizer)
        val_dataset = QADataset(val_data, tokenizer)
        
        # Određivanje batch size ovisno o modelu
        if model_name == 'XLM-RoBERTa':
            batch_size = 2  # Manji batch size za veći model
        else:
            batch_size = 4
        
        # Konfiguracija treniranja
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=1,
            eval_steps=5,
            save_steps=5,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
            fp16=True if self.device == "cuda" else False,
            gradient_accumulation_steps=4 if model_name == 'XLM-RoBERTa' else 2,
            gradient_checkpointing=True if model_name == 'XLM-RoBERTa' else False,
            report_to="none",
            remove_unused_columns=False
        )
        
        try:
            print(f"Početak treniranja modela {model_name}...")
            # Inicijalizacija trenera
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            # Treniranje modela
            train_result = trainer.train()
            
            print(f"Treniranje završeno. Spremanje modela...")
            # Spremanje najbolje verzije modela
            trainer.save_model(os.path.join(output_dir, f"best_{model_name}"))
            tokenizer.save_pretrained(os.path.join(output_dir, f"best_{model_name}"))
            
            print(f"Model uspješno spremljen u {output_dir}")
            return trainer
            
        except Exception as e:
            print(f"Greška tijekom treniranja modela {model_name}: {str(e)}")
            return None

    def cleanup(self):
        """Čišćenje memorije"""
        for model in self.models.values():
            model.cpu()
        self.models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()