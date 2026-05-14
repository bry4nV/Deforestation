import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def pipeline_persistencia(X_train, y_train, df_distritos_info, ruta_base, anios=None):
    """
    Modelo baseline de persistencia: predice que el futuro será igual
    al último valor observado (walk-forward con oracle).

    Genera:
    - _resultados.csv        (por distrito)
    - _departamento.csv
    - _global.csv
    - _config.json
    - _predicciones.csv
    - _ypred.npy
    """

    print("\n[INFO] Ejecutando persistencia (walk-forward)...")
    print("=" * 60)

    MODELO = "Persistencia_WF"

    n_distritos = X_train.shape[0]
    horizonte   = y_train.shape[1]

    y_pred_total = []
    registros    = []

    for i in range(n_distritos):
        history = X_train[i, :, 0].tolist()
        preds   = []

        for t in range(horizonte):
            yhat = history[-1]
            preds.append(yhat)
            history.append(float(y_train[i, t]))

        y_pred_total.append(preds)

        rmse_i, mae_i = calcular_metricas(y_train[i], preds)

        info = df_distritos_info.iloc[i]
        registros.append({
            "geocode":      info["geocode"],
            "departamento": info["departamento"],
            "distrito":     info["distrito"],
            "rmse":         round(rmse_i, 6),
            "mae":          round(mae_i, 6),
        })

    y_pred_total = np.array(y_pred_total)

    rmse_global, mae_global = calcular_metricas(y_train, y_pred_total)

    df_distrito = (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    departamentos = df_distritos_info["departamento"].values
    registros_dep = []
    for dep in np.unique(departamentos):
        mask = departamentos == dep
        rmse_dep, mae_dep = calcular_metricas(y_train[mask], y_pred_total[mask])
        registros_dep.append({
            "departamento": dep,
            "rmse":         round(rmse_dep, 6),
            "mae":          round(mae_dep, 6),
        })

    df_departamento = (
        pd.DataFrame(registros_dep)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    # Predicciones en formato largo
    pred_registros = []
    for i in range(n_distritos):
        info = df_distritos_info.iloc[i]
        for t in range(horizonte):
            y_true_val = float(y_train[i, t])
            y_pred_val = float(y_pred_total[i, t])
            error      = y_pred_val - y_true_val
            pred_registros.append({
                "modelo":        MODELO,
                "geocode":       info["geocode"],
                "departamento":  info["departamento"],
                "distrito":      info["distrito"],
                "horizonte":     t + 1,
                "anio":          anios[t] if anios is not None else None,
                "y_true":        y_true_val,
                "y_pred":        y_pred_val,
                "error":         error,
                "abs_error":     abs(error),
                "squared_error": error ** 2,
            })

    df_predicciones = pd.DataFrame(pred_registros)
    if anios is None:
        df_predicciones = df_predicciones.drop(columns=["anio"])

    config_dict = {
        "modelo": MODELO,
        "rmse":   round(rmse_global, 6),
        "mae":    round(mae_global, 6),
    }

    # Guardar
    ruta_dist = ruta_base.replace(".csv", "_resultados.csv")
    df_distrito.to_csv(ruta_dist, index=False)
    print(f"[OK] Por distrito:         {ruta_dist}")

    ruta_dep = ruta_base.replace(".csv", "_departamento.csv")
    df_departamento.to_csv(ruta_dep, index=False)
    print(f"[OK] Por departamento:     {ruta_dep}")

    ruta_global = ruta_base.replace(".csv", "_global.csv")
    pd.DataFrame([{
        "modelo": MODELO,
        "rmse":   round(rmse_global, 6),
        "mae":    round(mae_global, 6),
    }]).to_csv(ruta_global, index=False)
    print(f"[OK] Métricas globales:    {ruta_global}")

    ruta_config = ruta_base.replace(".csv", "_config.json")
    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"[OK] Config:               {ruta_config}")

    ruta_pred = ruta_base.replace(".csv", "_predicciones.csv")
    df_predicciones.to_csv(ruta_pred, index=False)
    print(f"[OK] Predicciones largas:  {ruta_pred}")

    ruta_ypred = ruta_base.replace(".csv", "_ypred.npy")
    np.save(ruta_ypred, y_pred_total)
    print(f"[OK] y_pred:               {ruta_ypred}")

    print(f"\n[OK] Resultado final Persistencia:")
    print(f"     Modelo: {MODELO}")
    print(f"     RMSE:   {rmse_global:.6f}")
    print(f"     MAE:    {mae_global:.6f}")

    return {
        "modelo":          MODELO,
        "rmse":            rmse_global,
        "mae":             mae_global,
        "y_pred":          y_pred_total,
        "config":          config_dict,
        "df_predicciones": df_predicciones,
        "df_departamento": df_departamento,
        "df_distrito":     df_distrito,
    }
