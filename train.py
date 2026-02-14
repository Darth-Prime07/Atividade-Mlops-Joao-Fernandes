import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from src.data import train_test_split_diamonds
from src.model import build_model
from src.evaluate import regression_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # pasta local para salvar o modelo para o Streamlit
    os.makedirs("models", exist_ok=True)
    
    # Configuração do MLflow (Certifique-se que o servidor está rodando)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("diamond_price_experiment")
    
    # 1. Carregar Dados
    X_train, X_test, y_train, y_test = train_test_split_diamonds(
        test_size=args.test_size
    )
    
    # 2. Construir Pipeline
    pipeline = build_model(df_sample=X_train, max_depth=args.max_depth)
    
    with mlflow.start_run():
        # Log Parâmetros
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        
        # 3. Treino (O pipeline cuida do pré-processamento)
        pipeline.fit(X_train, y_train)
        
        # 4. Predição
        y_pred = pipeline.predict(X_test)
        
        # 5. Métricas
        metrics = regression_metrics(y_test, y_pred)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
        # 6. Log do Modelo (MLflow)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            pipeline,
            "diamond_price_model",
            signature=signature,
        )
        
        # 7. Salvar localmente para o App
        joblib.dump(pipeline, "models/diamond_price_model.joblib")
        
        print("Treino concluído.")
        print(f"MAE:  {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2:   {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()