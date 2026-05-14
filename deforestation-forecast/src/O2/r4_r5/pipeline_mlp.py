import ast
import json
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from O2.config import SEMILLA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(seed=SEMILLA):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return rmse, mae


def diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test):
    gap_rmse = rmse_test - rmse_train
    gap_mae = mae_test - mae_train

    ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else np.nan
    ratio_mae = mae_test / mae_train if mae_train > 0 else np.nan

    return {
        "gap_rmse": round(float(gap_rmse), 6),
        "gap_mae": round(float(gap_mae), 6),
        "ratio_rmse": round(float(ratio_rmse), 6),
        "ratio_mae": round(float(ratio_mae), 6),
    }


def obtener_activacion(nombre):
    nombre = nombre.lower()

    if nombre == "relu":
        return nn.ReLU()
    elif nombre == "tanh":
        return nn.Tanh()
    elif nombre == "sigmoid":
        return nn.Sigmoid()
    elif nombre == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif nombre == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Función de activación no soportada: {nombre}")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, activation):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(obtener_activacion(activation))
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def parse_hidden_sizes(hidden_sizes):
    if isinstance(hidden_sizes, str):
        return ast.literal_eval(hidden_sizes)
    return hidden_sizes


def preparar_X_mlp(X_tensor):
    return X_tensor.reshape(X_tensor.shape[0], -1)


def entrenar(
    X_train_t,
    y_train_t,
    hidden_sizes,
    dropout,
    activation,
    epochs,
    batch_size,
    lr,
    seed=SEMILLA,
):

    fijar_semilla(seed)

    X = preparar_X_mlp(X_train_t).float()
    y = y_train_t.float()

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = MLP(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses    = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        train_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1} | Train={train_loss:.6f}")

    return model, train_losses


def evaluar(model, X_tensor, y_tensor):
    model.eval()

    X = preparar_X_mlp(X_tensor).float().to(DEVICE)
    y_true = y_tensor.detach().cpu().numpy()

    with torch.no_grad():
        preds = model(X).detach().cpu().numpy()

    rmse, mae = calcular_metricas(y_true, preds)

    return rmse, mae


def evaluar_geografico(
    model,
    series,
    df_distritos_info,
    window_size,
    tamanio_entrenamiento,
):
    model.eval()
    horizonte    = series.shape[1] - tamanio_entrenamiento
    y_pred_total = []
    registros    = []

    for i in range(series.shape[0]):
        history = series[i, :tamanio_entrenamiento].tolist()
        y_true_i = series[i, tamanio_entrenamiento:]
        preds = []

        for t in range(horizonte):
            x = torch.tensor(
                np.array(history[-window_size:], dtype=np.float32)
            ).unsqueeze(0)

            x = preparar_X_mlp(x).float().to(DEVICE)

            with torch.no_grad():
                yhat = model(x).item()

            preds.append(yhat)
            history.append(float(y_true_i[t]))

        y_pred_total.append(preds)

        preds_arr = np.array(preds)
        rmse_i, mae_i = calcular_metricas(y_true_i, preds_arr)

        info = df_distritos_info.iloc[i]

        registros.append({
            "geocode": info["geocode"],
            "departamento": info["departamento"],
            "distrito": info["distrito"],
            "rmse": round(rmse_i, 6),
            "mae": round(mae_i, 6),
        })

    y_pred_total = np.array(y_pred_total)
    y_true_total = series[:, tamanio_entrenamiento:]

    rmse_global, mae_global = calcular_metricas(y_true_total, y_pred_total)

    df_distrito = (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    departamentos = df_distritos_info["departamento"].values
    registros_dep = []

    for dep in np.unique(departamentos):
        mask = departamentos == dep
        y_t = y_true_total[mask]
        y_p = y_pred_total[mask]

        rmse_dep, mae_dep = calcular_metricas(y_t, y_p)

        registros_dep.append({
            "departamento": dep,
            "rmse": round(rmse_dep, 6),
            "mae": round(mae_dep, 6),
        })

    df_departamento = (
        pd.DataFrame(registros_dep)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    return df_distrito, df_departamento, rmse_global, mae_global, y_pred_total


def construir_df_predicciones(
    modelo_nombre,
    y_true_total,
    y_pred_total,
    df_distritos_info,
    anios_test=None,
):
    registros = []

    for i in range(y_true_total.shape[0]):
        info = df_distritos_info.iloc[i]

        for j in range(y_true_total.shape[1]):
            y_true = float(y_true_total[i, j])
            y_pred = float(y_pred_total[i, j])
            error = y_pred - y_true

            registro = {
                "modelo": modelo_nombre,
                "geocode": info["geocode"],
                "departamento": info["departamento"],
                "distrito": info["distrito"],
                "horizonte": j + 1,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error ** 2,
            }

            if anios_test is not None:
                registro["anio"] = anios_test[j]

            registros.append(registro)

    return pd.DataFrame(registros)


def graficar_curva(train_losses, nombre, ruta_png):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train MSE",
        linewidth=1.5,
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Curva de aprendizaje - {nombre}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=120)
    plt.close(fig)


# ============================================================
# FASE 1: Grid search exploratorio GPU
# ============================================================

def pipeline_mlp(
    dataset_dl,
    ruta_base,
    epochs_values,
    lr_values,
    batch_size_values,
    hidden_sizes_values,
    dropout_values,
    activation_values,
):
    """
    Esta fase sirve para analizar configuraciones.

    Genera:
    - _resultados.csv
    - _top5_configuraciones.csv
    - _mejores_por_ventana.csv
    """

    print("\n[INFO] Pipeline MLP — búsqueda exploratoria de hiperparámetros")
    print("=" * 70)
    print(f"[INFO] Device usado: {DEVICE}")

    if not dataset_dl:
        raise RuntimeError("Pipeline MLP: dataset_dl vacío, no hay ventanas válidas.")

    grid = list(product(
        dataset_dl.items(),
        hidden_sizes_values,
        dropout_values,
        activation_values,
        epochs_values,
        lr_values,
        batch_size_values,
    ))

    print(f"[INFO] Combinaciones totales: {len(grid)}")

    resultados = []

    for idx, ((w, data), hidden_sizes, dropout, activation, epochs, lr, batch_size) in enumerate(grid, start=1):
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        hidden_sizes = parse_hidden_sizes(hidden_sizes)

        model, _ = entrenar(
            X_train_t=X_train,
            y_train_t=y_train,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=SEMILLA,
        )

        rmse_train, mae_train = evaluar(model, X_train, y_train)
        rmse_test, mae_test = evaluar(model, X_test, y_test)

        diag = diagnosticar_ajuste(
            rmse_train=rmse_train,
            mae_train=mae_train,
            rmse_test=rmse_test,
            mae_test=mae_test,
        )

        h_str = "x".join(map(str, hidden_sizes))

        nombre = (
            f"MLP_w{w}_h{h_str}_act{activation}"
            f"_d{dropout}_e{epochs}_lr{lr}_b{batch_size}"
        )

        print(
            f"  [{idx}/{len(grid)}] {nombre}  "
            f"RMSE_test={rmse_test:.4f}  MAE_test={mae_test:.4f}"
        )

        fila = {
            "modelo": nombre,
            "window_size": int(w),
            "hidden_sizes": str(hidden_sizes),
            "activation": activation,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "rmse_train": round(rmse_train, 6),
            "mae_train": round(mae_train, 6),
            "rmse_test": round(rmse_test, 6),
            "mae_test": round(mae_test, 6),
            "gap_rmse": diag["gap_rmse"],
            "gap_mae": diag["gap_mae"],
            "ratio_rmse": diag["ratio_rmse"],
            "ratio_mae": diag["ratio_mae"],
        }

        resultados.append(fila)

    df = (
        pd.DataFrame(resultados)
        .sort_values(["rmse_test", "mae_test", "gap_rmse"])
        .reset_index(drop=True)
    )

    ruta_resultados = ruta_base.replace(".csv", "_resultados.csv")
    df.to_csv(ruta_resultados, index=False)
    print(f"\n[OK] Resultados completos guardados: {ruta_resultados}")

    df_top5 = df.head(5).copy()

    ruta_top5 = ruta_base.replace(".csv", "_top5_configuraciones.csv")
    df_top5.to_csv(ruta_top5, index=False)
    print(f"[OK] Top 5 configuraciones guardadas: {ruta_top5}")

    df_mejores_por_ventana = (
        df.groupby("window_size", sort=True)
        .first()
        .reset_index()
    )

    ruta_mejores_ventana = ruta_base.replace(".csv", "_mejores_por_ventana.csv")
    df_mejores_por_ventana.to_csv(ruta_mejores_ventana, index=False)
    print(f"[OK] Mejores por ventana guardado: {ruta_mejores_ventana}")

    top1 = df.iloc[0]
    print("\n[OK] Top 1 exploratorio (revisar _resultados.csv para elegir config final):")
    print(f"     Modelo:    {top1['modelo']}")
    print(f"     RMSE_test: {top1['rmse_test']}")
    print(f"     MAE_test:  {top1['mae_test']}")
    print(f"     gap_rmse:  {top1['gap_rmse']}")
    print(f"     ratio_rmse:{top1['ratio_rmse']}")
    print(f"     Device:    {DEVICE}")

    return {
        "grid_resultados": df,
        "top5": df_top5,
        "mejores_por_ventana": df_mejores_por_ventana,
    }


# ============================================================
# FASE 2: Entrenamiento final con configuración elegida GPU
# ============================================================

def entrenar_config_final_mlp(
    dataset_dl,
    final_config,
    ruta_base,
    series,
    df_distritos_info,
    tamanio_entrenamiento,
    anios=None,
):
    """
    Esta función se usa después de revisar:
    - _resultados.csv
    - _top5_configuraciones.csv
    - _mejores_por_ventana.csv

    Toma UNA configuración final y genera:
    - _final_model.pth
    - _final_curva.png
    - _final_global.csv
    - _final_departamento.csv
    - _final_distrito.csv
    - _final_predicciones.csv
    - _final_ypred.npy
    """

    print("\n[INFO] Entrenamiento final MLP")
    print("=" * 70)
    print(f"[INFO] Device usado: {DEVICE}")

    final_config = dict(final_config)

    window_size = int(final_config["window_size"])
    hidden_sizes = parse_hidden_sizes(final_config["hidden_sizes"])
    activation = final_config["activation"]
    dropout = float(final_config["dropout"])
    epochs = int(final_config["epochs"])
    lr = float(final_config["lr"])
    batch_size = int(final_config["batch_size"])

    if window_size not in dataset_dl:
        raise ValueError(f"window_size={window_size} no existe en dataset_dl.")

    data = dataset_dl[window_size]
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    nombre = (
        f"MLP_FINAL_w{window_size}_h{'x'.join(map(str, hidden_sizes))}"
        f"_act{activation}_d{dropout}_e{epochs}_lr{lr}_b{batch_size}"
    )

    seed_final = SEMILLA

    model, train_losses = entrenar(
        X_train_t=X_train,
        y_train_t=y_train,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed_final,
    )

    rmse_train, mae_train = evaluar(model, X_train, y_train)
    rmse_test, mae_test = evaluar(model, X_test, y_test)

    diag = diagnosticar_ajuste(
        rmse_train=rmse_train,
        mae_train=mae_train,
        rmse_test=rmse_test,
        mae_test=mae_test,
    )

    final_row = {
        "modelo": nombre,
        "window_size": window_size,
        "hidden_sizes": str(hidden_sizes),
        "activation": activation,
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed_final,
        "rmse_train": round(rmse_train, 6),
        "mae_train": round(mae_train, 6),
        "rmse_test": round(rmse_test, 6),
        "mae_test": round(mae_test, 6),
        "gap_rmse": diag["gap_rmse"],
        "gap_mae": diag["gap_mae"],
        "ratio_rmse": diag["ratio_rmse"],
        "ratio_mae": diag["ratio_mae"],
    }

    df_distrito, df_departamento, rmse_wf, mae_wf, y_pred_wf = evaluar_geografico(
        model=model,
        series=series,
        df_distritos_info=df_distritos_info,
        window_size=window_size,
        tamanio_entrenamiento=tamanio_entrenamiento,
    )

    y_true_total = series[:, tamanio_entrenamiento:]

    anios_test = None
    if anios is not None:
        anios_test = anios[tamanio_entrenamiento:]

    df_predicciones = construir_df_predicciones(
        modelo_nombre=nombre,
        y_true_total=y_true_total,
        y_pred_total=y_pred_wf,
        df_distritos_info=df_distritos_info,
        anios_test=anios_test,
    )

    ruta_model = ruta_base.replace(".csv", "_final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": final_row,
            "train_losses": train_losses,
            "seed": seed_final,
            "model_type": "MLP",
            "device_entrenamiento": str(DEVICE),
        },
        ruta_model,
    )

    ruta_curva = ruta_base.replace(".csv", "_final_curva.png")
    graficar_curva(train_losses, nombre, ruta_curva)

    ruta_global = ruta_base.replace(".csv", "_final_global.csv")
    pd.DataFrame([{
        "modelo": nombre,
        "rmse": round(rmse_wf, 6),
        "mae": round(mae_wf, 6),
        "rmse_train": round(rmse_train, 6),
        "mae_train": round(mae_train, 6),
        "rmse_test_directo": round(rmse_test, 6),
        "mae_test_directo": round(mae_test, 6),
        "gap_rmse": diag["gap_rmse"],
        "gap_mae": diag["gap_mae"],
        "ratio_rmse": diag["ratio_rmse"],
        "ratio_mae": diag["ratio_mae"],
    }]).to_csv(ruta_global, index=False)

    ruta_config = ruta_base.replace(".csv", "_final_config.json")
    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(final_row, f, indent=4, ensure_ascii=False)

    ruta_dist = ruta_base.replace(".csv", "_final_distrito.csv")
    df_distrito.to_csv(ruta_dist, index=False)

    ruta_dep = ruta_base.replace(".csv", "_final_departamento.csv")
    df_departamento.to_csv(ruta_dep, index=False)

    ruta_pred = ruta_base.replace(".csv", "_final_predicciones.csv")
    df_predicciones.to_csv(ruta_pred, index=False)

    ruta_ypred = ruta_base.replace(".csv", "_final_ypred.npy")
    np.save(ruta_ypred, y_pred_wf)

    print(f"[OK] Modelo final guardado:        {ruta_model}")
    print(f"[OK] Curva final guardada:         {ruta_curva}")
    print(f"[OK] Config final guardada:        {ruta_config}")
    print(f"[OK] Métricas globales:            {ruta_global}")
    print(f"[OK] Métricas por distrito:        {ruta_dist}")
    print(f"[OK] Métricas por departamento:    {ruta_dep}")
    print(f"[OK] Predicciones finales CSV:     {ruta_pred}")
    print(f"[OK] Predicciones finales NPY:     {ruta_ypred}")

    print("\n[OK] Resultado final MLP:")
    print(f"     Modelo:  {nombre}")
    print(f"     RMSE_wf: {rmse_wf:.6f}")
    print(f"     MAE_wf:  {mae_wf:.6f}")
    print(f"     Device:  {DEVICE}")

    return {
        "modelo": nombre,
        "rmse": rmse_wf,
        "mae": mae_wf,
        "y_pred": y_pred_wf,
        "config": final_row,
        "df_predicciones": df_predicciones,
        "df_departamento": df_departamento,
        "df_distrito": df_distrito,
    }