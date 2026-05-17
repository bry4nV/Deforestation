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
    elif nombre == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif nombre == "tanh":
        return nn.Tanh()
    elif nombre == "elu":
        return nn.ELU()
    elif nombre == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Función de activación no soportada: {nombre}")


def parse_num_channels(num_channels):
    if isinstance(num_channels, str):
        return ast.literal_eval(num_channels)
    return num_channels


def preparar_X_tcn(X_tensor):
    """Convierte (batch, window_size) o (batch, window_size, 1) a (batch, 1, window_size)."""
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(-1)

    return X_tensor.permute(0, 2, 1).float()


class TCNBlock(nn.Module):
    """
    Bloque residual con dos convoluciones causales dilatadas.

    La dilation crece exponencialmente por nivel: nivel 0 → dil=1, nivel 1 → dil=2, etc.
    El padding causal (solo por la izquierda) garantiza que no haya fuga de información futura.
    Se recortan los `padding` elementos finales del output para mantener la longitud temporal.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, activation):
        super().__init__()

        self.chomp_size = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.chomp_size,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.chomp_size,
        )

        self.act1 = obtener_activacion(activation)
        self.act2 = obtener_activacion(activation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # Proyección 1×1 si los canales de entrada y salida difieren
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.act_out = obtener_activacion(activation)

    def _chomp(self, x):
        """Elimina los elementos finales añadidos por el padding causal."""
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self._chomp(self.conv1(x))
        out = self.drop1(self.act1(out))

        out = self._chomp(self.conv2(out))
        out = self.drop2(self.act2(out))

        return self.act_out(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (Bai et al., 2018).

    Apila TCNBlocks con dilation=2^i en el nivel i. El campo receptivo efectivo
    crece exponencialmente con la profundidad sin aumentar el número de parámetros
    por capa. Toma el último paso temporal para la predicción final (one-step-ahead).

    Input:  (batch, 1, window_size)
    Output: (batch, 1)
    """

    def __init__(self, input_channels, num_channels, kernel_size, dropout, activation):
        super().__init__()

        layers = []
        in_ch = input_channels

        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout, activation)
            )
            in_ch = out_ch

        self.red = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (batch, 1, window_size)
        out = self.red(x)      # (batch, out_channels, window_size)
        out = out[:, :, -1]    # último paso temporal: (batch, out_channels)
        return self.fc(out)    # (batch, 1)


def entrenar(
    X_train_t,
    y_train_t,
    num_channels,
    kernel_size,
    dropout,
    activation,
    epochs,
    batch_size,
    lr,
    seed=SEMILLA,
):
    fijar_semilla(seed)

    X = preparar_X_tcn(X_train_t)
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

    model = TCN(
        input_channels=X.shape[1],
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        activation=activation,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

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

    X = preparar_X_tcn(X_tensor).to(DEVICE)
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

    horizonte = series.shape[1] - tamanio_entrenamiento
    y_pred_total = []
    registros = []

    for i in range(series.shape[0]):
        history = series[i, :tamanio_entrenamiento].tolist()
        y_true_i = series[i, tamanio_entrenamiento:]
        preds = []

        for t in range(horizonte):
            x = torch.tensor(
                np.array(history[-window_size:], dtype=np.float32)
            ).unsqueeze(0).unsqueeze(-1)

            # De (1, window_size, 1) a (1, 1, window_size)
            x = preparar_X_tcn(x).to(DEVICE)

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
# FASE 1: Grid search exploratorio TCN
# ============================================================

def pipeline_tcn(
    dataset_dl,
    ruta_base,
    epochs_values,
    lr_values,
    batch_size_values,
    num_channels_values,
    kernel_size_values,
    dropout_values,
    activation_values,
):
    """
    Fase 1: búsqueda exploratoria de hiperparámetros.

    Genera:
    - _resultados.csv
    - _top5_configuraciones.csv
    - _mejores_por_ventana.csv

    No guarda modelo.
    No guarda curva.
    No hace evaluación geográfica.
    """

    print("\n[INFO] Pipeline TCN — búsqueda exploratoria de hiperparámetros")
    print("=" * 70)
    print(f"[INFO] Device usado: {DEVICE}")

    if not dataset_dl:
        raise RuntimeError("Pipeline TCN: dataset_dl vacío, no hay ventanas válidas.")

    grid = list(product(
        dataset_dl.items(),
        num_channels_values,
        kernel_size_values,
        dropout_values,
        activation_values,
        epochs_values,
        lr_values,
        batch_size_values,
    ))

    print(f"[INFO] Combinaciones totales: {len(grid)}")

    resultados = []

    for idx, (
        (w, data),
        num_channels,
        kernel_size,
        dropout,
        activation,
        epochs,
        lr,
        batch_size,
    ) in enumerate(grid, start=1):

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        num_channels = parse_num_channels(num_channels)

        model, _ = entrenar(
            X_train_t=X_train,
            y_train_t=y_train,
            num_channels=num_channels,
            kernel_size=kernel_size,
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

        ch_str = "x".join(map(str, num_channels))

        nombre = (
            f"TCN_w{w}_c{ch_str}_k{kernel_size}_act{activation}"
            f"_d{dropout}_e{epochs}_lr{lr}_b{batch_size}"
        )

        print(
            f"  [{idx}/{len(grid)}] {nombre}  "
            f"RMSE_test={rmse_test:.4f}  MAE_test={mae_test:.4f}"
        )

        fila = {
            "modelo": nombre,
            "window_size": int(w),
            "num_channels": str(num_channels),
            "kernel_size": int(kernel_size),
            "activation": activation,
            "dropout": float(dropout),
            "epochs": int(epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
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

    if not resultados:
        raise RuntimeError("Pipeline TCN: no se entrenó ninguna configuración válida.")

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

    print("\n[OK] Top 1 exploratorio. Revisar CSVs antes de elegir config final:")
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
# FASE 2: Entrenamiento final TCN
# ============================================================

def entrenar_config_final_tcn(
    dataset_dl,
    final_config,
    ruta_base,
    series,
    df_distritos_info,
    tamanio_entrenamiento,
    anios=None,
):
    """
    Fase 2: entrenamiento final con UNA configuración elegida.

    Genera:
    - _final_model.pth
    - _final_curva.png
    - _final_config.json
    - _final_global.csv
    - _final_distrito.csv
    - _final_departamento.csv
    - _final_predicciones.csv
    - _final_ypred.npy
    """

    print("\n[INFO] Entrenamiento final TCN")
    print("=" * 70)
    print(f"[INFO] Device usado: {DEVICE}")

    final_config = dict(final_config)

    window_size  = int(final_config["window_size"])
    num_channels = parse_num_channels(final_config["num_channels"])
    kernel_size  = int(final_config["kernel_size"])
    activation   = final_config["activation"]
    dropout      = float(final_config["dropout"])
    epochs       = int(final_config["epochs"])
    lr           = float(final_config["lr"])
    batch_size   = int(final_config["batch_size"])

    if window_size not in dataset_dl:
        raise ValueError(f"window_size={window_size} no existe en dataset_dl.")

    data = dataset_dl[window_size]
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    ch_str = "x".join(map(str, num_channels))

    nombre = (
        f"TCN_FINAL_w{window_size}_c{ch_str}_k{kernel_size}_act{activation}"
        f"_d{dropout}_e{epochs}_lr{lr}_b{batch_size}"
    )

    model, train_losses = entrenar(
        X_train_t=X_train,
        y_train_t=y_train,
        num_channels=num_channels,
        kernel_size=kernel_size,
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

    final_row = {
        "modelo": nombre,
        "window_size": window_size,
        "num_channels": str(num_channels),
        "kernel_size": kernel_size,
        "activation": activation,
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": SEMILLA,
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
            "seed": SEMILLA,
            "model_type": "TCN",
            "device_entrenamiento": str(DEVICE),
        },
        ruta_model,
    )

    ruta_curva = ruta_base.replace(".csv", "_final_curva.png")
    graficar_curva(train_losses, nombre, ruta_curva)

    ruta_config = ruta_base.replace(".csv", "_final_config.json")
    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(final_row, f, indent=4, ensure_ascii=False)

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

    print("\n[OK] Resultado final TCN:")
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
