import data_preparation
import models
import evaluation

def main():
    # Inicijalizacija komponenti
    data_prep = data_preparation.DataPreparator()
    model_manager = models.ModelManager()
    evaluator = evaluation.Evaluator()
    
    # Priprema podataka
    train_data, val_data, test_data = data_prep.prepare_tourism_data('tourism_guides.csv')
    
    # Inicijalizacija modela
    model_manager.initialize_models()
    
    # Treniranje i evaluacija svakog modela
    results = {}
    for model_name in model_manager.models:
        print(f"\nTreniranje i evaluacija modela: {model_name}")
        
        # Treniranje
        trainer = model_manager.train_model(
            model_name,
            train_data,
            val_data,
            f'./output/{model_name}'
        )
        
        # Evaluacija
        model_results = evaluator.evaluate_model(
            model_manager.models[model_name],
            model_manager.tokenizers[model_name],
            test_data
        )
        
        results[model_name] = model_results
    
    # Ispis rezultata
    print("\nRezultati evaluacije:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for metric, score in model_results.items():
            print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()