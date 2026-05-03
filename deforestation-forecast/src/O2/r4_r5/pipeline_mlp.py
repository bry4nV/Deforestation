import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MLP(nn.Module):
    """Multilayer Perceptron para pronóstico univariado"""

    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def entrenar_mlp(model, dataloader, epochs, lr):
    """Entrena el modelo MLP"""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        epoch_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in dataloader:

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model


def evaluar_mlp(model, X_tensor, y_tensor, set_name="Test"):
    """Evalúa el modelo MLP y retorna métricas"""

    model.eval()
    X = X_tensor.view(X_tensor.shape[0], -1)

    with torch.no_grad():
        preds = model(X).numpy()

    y_true = y_tensor.numpy()

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)

    print(f"[RESULT] {set_name} RMSE: {rmse:.4f}")
    print(f"[RESULT] {set_name} MAE: {mae:.4f}")

    return rmse, mae, preds


def pipeline_mlp(
    datasets_dl,
    df_distritos_info,
    window_size,
    epochs=50,
    lr=0.01,
    batch_size=32,
    hidden_sizes=[64, 32],
    ruta_modelo_mlp=None,
    exportar=True
):
    """
    Pipeline completo de MLP: entrenamiento + evaluación.

    Args:
        datasets_dl: dict {window_size: {"train": (X_train, y_train), "test": (X_test, y_test)}}
        df_distritos_info: DataFrame con info de distritos (geocode, departamento, distrito)
        window_size: tamaño de ventana a usar
        epochs: número de épocas de entrenamiento
        lr: learning rate
        batch_size: tamaño de batch
        hidden_sizes: lista de tamaños de capas ocultas
        ruta_modelo_mlp: ruta donde guardar resultados CSV
        exportar: si guardar resultados a CSV

    Returns:
        dict con métricas del modelo
    """

    print(f"\n[INFO] Ejecutando pipeline MLP (window={window_size})...")

    # =====================================================================
    # Cargar datasets
    # =====================================================================
    X_train_tensor, y_train_tensor = datasets_dl[window_size]["train"]
    X_test_tensor, y_test_tensor = datasets_dl[window_size]["test"]

    print(f"[INFO] Dataset shapes:")
    print(f"  X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")
    print(f"  X_test: {X_test_tensor.shape}, y_test: {y_test_tensor.shape}")

    # =====================================================================
    # Preparar DataLoader para entrenamiento
    # =====================================================================
    X_train_flat = X_train_tensor.view(X_train_tensor.shape[0], -1)
    dataset = TensorDataset(X_train_flat, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train_flat.shape[1]

    # =====================================================================
    # Crear modelo
    # =====================================================================
    print(f"\n[INFO] Creando MLP: input={input_size}, hidden={hidden_sizes}, output=1")
    model = MLP(input_size, hidden_sizes=hidden_sizes)

    # =====================================================================
    # Entrenar
    # =====================================================================
    print(f"\n[INFO] Entrenando por {epochs} épocas (lr={lr}, batch={batch_size})...")
    model = entrenar_mlp(model, dataloader, epochs, lr)

    # =====================================================================
    # Evaluar en TRAIN
    # =====================================================================
    print(f"\n[INFO] Evaluando en TRAIN...")
    rmse_train, mae_train, preds_train = evaluar_mlp(model, X_train_tensor, y_train_tensor, "Train")

    # =====================================================================
    # Evaluar en TEST
    # =====================================================================
    print(f"\n[INFO] Evaluando en TEST...")
    rmse_test, mae_test, preds_test = evaluar_mlp(model, X_test_tensor, y_test_tensor, "Test")

    # =====================================================================
    # Exportar resultados (similar a ARIMA y Persistencia)
    # =====================================================================
    if exportar and ruta_modelo_mlp:

        # Por ahora exportamos resumen global
        df_global = pd.DataFrame([{
            "modelo": f"MLP_w{window_size}",
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "rmse_train": rmse_train,
            "mae_train": mae_train,
            "rmse_test": rmse_test,
            "mae_test": mae_test
        }])

        df_global.to_csv(ruta_modelo_mlp, index=False)
        print(f"\n[OK] Resultados guardados: {ruta_modelo_mlp}")

    # =====================================================================
    # Retornar en formato estándar
    # =====================================================================
    resultado = {
        "modelo": f"MLP_w{window_size}",
        "window_size": window_size,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "preds_train": preds_train,
        "preds_test": preds_test,
        "model": model
    }

    return resultado
