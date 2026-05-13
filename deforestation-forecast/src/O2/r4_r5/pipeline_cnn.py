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


def fijar_semilla():
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
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
    else:
        raise ValueError(f"Función de activación no soportada: {nombre}")


class CNN1D(nn.Module):
    def __init__(
        self,
        input_channels,
        window_size,
        conv_channels,
        kernel_size,
        dropout,
        activation,
        dense_size,
    ):
        super().__init__()

        if kernel_size > window_size:
            raise ValueError(
                f"kernel_size={kernel_size} no puede ser mayor que window_size={window_size}"
            )

        layers = []
        prev_channels = input_channels

        for out_channels in conv_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same"
                )
            )
            layers.append(obtener_activacion(activation))
            layers.append(nn.Dropout(dropout))
            prev_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Con padding="same", la longitud temporal de salida se mantiene igual a window_size.
        conv_output_size = conv_channels[-1] * window_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, dense_size),
            obtener_activacion(activation),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def entrenar(
    X_train_t,
    y_train_t,
    conv_channels,
    kernel_size,
    dropout,
    activation,
    dense_size,
    epochs,
    batch_size,
    lr,
):
    """
    X_train_t esperado:
        shape = (n_muestras, window_size, 1)
        igual que LSTM.

    Para CNN se transforma internamente a:
        shape = (n_muestras, 1, window_size)
    """

    X_train_cnn = X_train_t.permute(0, 2, 1)

    dataloader = DataLoader(
        TensorDataset(X_train_cnn, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    model = CNN1D(
        input_channels=X_train_cnn.shape[1],
        window_size=X_train_cnn.shape[2],
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        activation=activation,
        dense_size=dense_size,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        train_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Train={train_loss:.4f}")

    return model, train_losses


def evaluar(model, X_tensor, y_tensor):
    """
    X_tensor esperado:
        shape = (n_muestras, window_size, 1)

    Para CNN:
        shape = (n_muestras, 1, window_size)
    """

    model.eval()

    X_cnn = X_tensor.permute(0, 2, 1)

    with torch.no_grad():
        preds = model(X_cnn).cpu().numpy()

    y_true = y_tensor.cpu().numpy()

    rmse, mae = calcular_metricas(y_true, preds)

    return rmse, mae


def evaluar_geografico(model, series, df_distritos_info, window_size, tamanio_entrenamiento):
    """
    Walk-forward evaluation:
    - history empieza con los años de entrenamiento.
    - para predecir cada año del test se usa la ventana más reciente.
    - después de predecir, se incorpora el valor real observado.
    """

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
                np.array(history[-window_size:], dtype=np.float32)[np.newaxis, np.newaxis, :]
            )
            # shape CNN: (1, 1, window_size)

            with torch.no_grad():
                yhat = model(x).item()

            preds.append(yhat)

            # Walk-forward one-step-ahead:
            # se agrega el valor real observado del año que acaba de predecirse
            history.append(float(y_true_i[t]))

        y_pred_total.append(preds)

        preds_arr = np.array(preds)
        rmse_i, mae_i = calcular_metricas(y_true_i, preds_arr)

        info = df_distritos_info.iloc[i]
        registros.append({
            "geocode":      info["geocode"],
            "departamento": info["departamento"],
            "distrito":     info["distrito"],
            "rmse":         round(rmse_i, 6),
            "mae":          round(mae_i, 6),
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


def graficar_curva(train_losses, nombre, ruta_png):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train MSE",
        linewidth=1.5
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Curva de aprendizaje - {nombre}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=120)
    plt.close(fig)


def pipeline_cnn(
    dataset_dl,
    ruta_base,
    epochs_values,
    lr_values,
    batch_size_values,
    conv_channels_values,
    kernel_size_values,
    dropout_values,
    activation_values,
    dense_size_values,
    series,
    df_distritos_info,
    tamanio_entrenamiento,
):
    print("\n[INFO] Pipeline CNN — búsqueda de hiperparámetros")
    print("=" * 60)

    if not dataset_dl:
        raise RuntimeError("Pipeline CNN: dataset_dl vacío, no hay ventanas válidas.")

    fijar_semilla()

    grid = list(product(
        dataset_dl.items(),
        conv_channels_values,
        kernel_size_values,
        dropout_values,
        activation_values,
        dense_size_values,
        epochs_values,
        lr_values,
        batch_size_values,
    ))

    print(f"[INFO] Combinaciones totales: {len(grid)}")

    resultados = []
    mejor_rmse = float("inf")
    mejor_modelo = None
    mejor_fila = None
    mejor_losses = None

    for (
        (w, data),
        conv_channels,
        kernel_size,
        dropout,
        activation,
        dense_size,
        epochs,
        lr,
        batch_size,
    ) in grid:

        # Saltar kernels mayores que la ventana.
        if kernel_size > int(w):
            continue

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        model, train_losses = entrenar(
            X_train,
            y_train,
            conv_channels,
            kernel_size,
            dropout,
            activation,
            dense_size,
            epochs,
            batch_size,
            lr,
        )

        rmse_train, mae_train = evaluar(model, X_train, y_train)
        rmse_test, mae_test = evaluar(model, X_test, y_test)

        diag = diagnosticar_ajuste(
            rmse_train,
            mae_train,
            rmse_test,
            mae_test
        )

        c_str = "x".join(map(str, conv_channels))
        nombre = (
            f"CNN_w{w}_c{c_str}_k{kernel_size}_act{activation}"
            f"_d{dropout}_dense{dense_size}_e{epochs}_lr{lr}_b{batch_size}"
        )

        print(
            f"  {nombre}  "
            f"RMSE_test={rmse_test:.4f}  MAE_test={mae_test:.4f}"
        )

        fila = {
            "modelo":       nombre,
            "window_size":  w,
            "conv_channels": str(conv_channels),
            "kernel_size":  kernel_size,
            "activation":   activation,
            "dropout":      dropout,
            "dense_size":   dense_size,
            "epochs":       epochs,
            "lr":           lr,
            "batch_size":   batch_size,
            "rmse_train":   round(rmse_train, 6),
            "mae_train":    round(mae_train,  6),
            "rmse_test":    round(rmse_test,  6),
            "mae_test":     round(mae_test,   6),
            "gap_rmse":     diag["gap_rmse"],
            "gap_mae":      diag["gap_mae"],
            "ratio_rmse":   diag["ratio_rmse"],
            "ratio_mae":    diag["ratio_mae"],
        }

        resultados.append(fila)

        if rmse_test < mejor_rmse:
            mejor_rmse = rmse_test
            mejor_modelo = model
            mejor_fila = fila
            mejor_losses = train_losses

    if not resultados:
        raise RuntimeError("Pipeline CNN: no se entrenó ninguna configuración válida.")

    df = pd.DataFrame(resultados).sort_values("rmse_test").reset_index(drop=True)

    ruta_csv = ruta_base.replace(".csv", "_resultados.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"\n[OK] Resultados CNN guardados: {ruta_csv}")

    ruta_pth = ruta_base.replace(".csv", "_mejor.pth")
    torch.save(
        {
            "model_state_dict": mejor_modelo.state_dict(),
            "config": mejor_fila
        },
        ruta_pth
    )
    print(f"[OK] Mejor modelo guardado:   {ruta_pth}")

    ruta_png = ruta_base.replace(".csv", "_mejor_curva.png")
    graficar_curva(mejor_losses, mejor_fila["modelo"], ruta_png)
    print(f"[OK] Curva de aprendizaje:    {ruta_png}")

    df_mejores_por_ventana = (
        pd.DataFrame(resultados)
        .sort_values("rmse_test")
        .groupby("window_size", sort=True)
        .first()
        .reset_index()
    )

    ruta_mejores = ruta_base.replace(".csv", "_mejores_por_ventana.csv")
    df_mejores_por_ventana.to_csv(ruta_mejores, index=False)
    print(f"[OK] Mejores por ventana:     {ruta_mejores}")

    df_distrito, df_departamento, rmse_wf, mae_wf, y_pred_wf = evaluar_geografico(
        mejor_modelo,
        series,
        df_distritos_info,
        int(mejor_fila["window_size"]),
        tamanio_entrenamiento,
    )

    ruta_dist = ruta_base.replace(".csv", "_mejor_distrito.csv")
    df_distrito.to_csv(ruta_dist, index=False)
    print(f"[OK] Por distrito:            {ruta_dist}")

    ruta_dep = ruta_base.replace(".csv", "_mejor_departamento.csv")
    df_departamento.to_csv(ruta_dep, index=False)
    print(f"[OK] Por departamento:        {ruta_dep}")

    ruta_global = ruta_base.replace(".csv", "_global.csv")
    pd.DataFrame([{
        "modelo": mejor_fila["modelo"],
        "rmse":   round(rmse_wf, 6),
        "mae":    round(mae_wf,  6),
    }]).to_csv(ruta_global, index=False)
    print(f"[OK] Métricas globales:       {ruta_global}")

    ruta_ypred = ruta_base.replace(".csv", "_mejor_ypred.npy")
    np.save(ruta_ypred, y_pred_wf)
    print(f"[OK] y_pred walk-forward:     {ruta_ypred}")

    print(
        f"[OK] Mejor config: {mejor_fila['modelo']}  "
        f"RMSE_wf={rmse_wf:.4f}  MAE_wf={mae_wf:.4f}"
    )

    return {
        "modelo": mejor_fila["modelo"],
        "rmse":   rmse_wf,
        "mae":    mae_wf,
        "y_pred": y_pred_wf,
    }