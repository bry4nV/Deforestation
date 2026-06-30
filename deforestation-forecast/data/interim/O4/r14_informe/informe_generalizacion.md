# Informe de generalización espacial — O4 / R14 (CNN1D extendido)

## 1. Métricas globales (en muestra vs. 20 zonas nuevas)

| modelo | n_distritos_en_muestra | n_predicciones_en_muestra | rmse_en_muestra | mae_en_muestra | n_distritos_generalizacion | n_predicciones_generalizacion | rmse_generalizacion | mae_generalizacion | diff_rmse_pct | diff_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 180 | 900 | 0.010912 | 0.00774 | 20 | 100 | 0.01129 | 0.007862 | 3.464 | 1.576 |


## 2. Tabla distrital completa (20 zonas)

| geocode | departamento | distrito | rmse | mae | sesgo | clasificacion |
| --- | --- | --- | --- | --- | --- | --- |
| 010113 | Amazonas | Mariscal Castilla | 0.00276 | 0.002664 | -0.00088960690395836 | menor error |
| 010101 | Amazonas | Chachapoyas | 0.003559 | 0.003341 | -0.0033406634918672603 | menor error |
| 010116 | Amazonas | Olleros | 0.005198 | 0.002989 | -0.0022258546336499 | menor error |
| 100702 | Huanuco | Cholon | 0.005588 | 0.005419 | -0.0032684500341826003 | menor error |
| 100603 | Huanuco | Hermilio Valdizan | 0.006507 | 0.005924 | -0.0020838796531734596 | menor error |
| 220103 | San Martin | Habana | 0.006599 | 0.005377 | 0.00018219276999686032 | menor error |
| 220903 | San Martin | Cacatachi | 0.006962 | 0.006093 | 0.0045749748361845 | menor error |
| 060805 | Cajamarca | Huabal | 0.007146 | 0.005367 | 0.00536699397424412 | error medio |
| 220104 | San Martin | Jepelacio | 0.007173 | 0.006261 | 0.00560102567036924 | error medio |
| 220105 | San Martin | Soritor | 0.007854 | 0.005215 | 0.0023831116169846186 | error medio |
| 160201 | Loreto | Yurimaguas | 0.007925 | 0.006019 | -0.0023571203815081804 | error medio |
| 080917 | Cusco | Manitea | 0.008132 | 0.007174 | -0.0065816797439576195 | error medio |
| 220406 | San Martin | Tingo De Saposoa | 0.008544 | 0.007049 | 0.00251788348661752 | error medio |
| 220801 | San Martin | Rioja | 0.008658 | 0.006272 | 0.0046029114983873395 | mayor error |
| 250303 | Ucayali | Curimana | 0.010984 | 0.008446 | -0.00262095813512796 | mayor error |
| 010506 | Amazonas | Inguilpata | 0.011701 | 0.008868 | -0.00838741548420054 | mayor error |
| 220709 | San Martin | Tingo De Ponasa | 0.014366 | 0.010745 | -0.0098566533383199 | mayor error |
| 060905 | Cajamarca | Namballe | 0.020564 | 0.016571 | -0.0019022471812170996 | mayor error |
| 220703 | San Martin | Caspisapa | 0.021004 | 0.015145 | 0.00279993676298772 | mayor error |
| 220509 | San Martin | Shanao | 0.023815 | 0.022294 | 0.01828689386815208 | mayor error |


## 3. Casos extremos

| geocode | departamento | distrito | rmse | mae | grupo |
| --- | --- | --- | --- | --- | --- |
| 010113 | Amazonas | Mariscal Castilla | 0.00276 | 0.002664 | menor error |
| 010101 | Amazonas | Chachapoyas | 0.003559 | 0.003341 | menor error |
| 010116 | Amazonas | Olleros | 0.005198 | 0.002989 | menor error |
| 220509 | San Martin | Shanao | 0.023815 | 0.022294 | mayor error |
| 220703 | San Martin | Caspisapa | 0.021004 | 0.015145 | mayor error |
| 060905 | Cajamarca | Namballe | 0.020564 | 0.016571 | mayor error |


## 4. Factores territoriales — análisis explicativo

_Dinámicas (pct_agropecuario, pct_anp) promediadas sobre el periodo de evaluación 2020–2024; estáticas con su valor por distrito. Mismo protocolo que la sección 6.5.5 del informe._

### 4.1 Correlaciones RMSE × factor (Pearson y Spearman)

| modelo | factor | n_zonas | r_pearson | p_pearson | r_spearman | p_spearman | media_grupo_menor_error | media_grupo_mayor_error | diferencia_grupos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | pct_agropecuario | 20 | 0.1266 | 0.5949 | 0.3203 | 0.1686 | 0.18399 | 0.43154 | 0.24755 |
| CNN | pct_anp | 20 | 0.3911 | 0.0881 | -0.1672 | 0.4811 | 0.02919 | 0.18036 | 0.15116 |
| CNN | densidad_carreteras_km_km2 | 20 | 0.3636 | 0.1151 | -0.0045 | 0.9849 | 0.39908 | 0.78407 | 0.38498 |
| CNN | densidad_rios_km_km2 | 20 | 0.2293 | 0.3307 | 0.2393 | 0.3097 | 0.03274 | 0.05225 | 0.01951 |
| CNN | elev_media_m | 20 | -0.3959 | 0.084 | -0.4962 | 0.0261 | 2833.03473 | 944.49173 | -1888.54299 |
| CNN | pendiente_media_deg | 20 | -0.0779 | 0.7442 | -0.1669 | 0.4818 | 18.91918 | 15.73216 | -3.18701 |


### 4.2 Perfil territorial promedio por grupo de error (terciles de RMSE)

| grupo_error | n_distritos | rmse_promedio | pct_agropecuario | pct_anp | densidad_carreteras_km_km2 | densidad_rios_km_km2 | elev_media_m | pendiente_media_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| menor error | 7 | 0.00531 | 0.35266 | 0.01251 | 0.42875 | 0.0203 | 1951.81043 | 15.45907 |
| error medio | 6 | 0.007796 | 0.47315 | 1e-05 | 0.39585 | 0.04393 | 1101.34227 | 15.87876 |
| mayor error | 7 | 0.01587 | 0.42283 | 0.09814 | 0.46919 | 0.03355 | 1036.08374 | 14.02442 |


### 4.3 Casos extremos con sus 6 variables locales

| geocode | departamento | distrito | grupo | rmse | pct_agropecuario | pct_anp | densidad_carreteras_km_km2 | densidad_rios_km_km2 | elev_media_m | pendiente_media_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 010113 | Amazonas | Mariscal Castilla | menor error | 0.00276 | 0.2201 | 0.01102 | 0.41978 | 0.04262 | 2905.48819 | 16.89758 |
| 010101 | Amazonas | Chachapoyas | menor error | 0.003559 | 0.24074 | 0.04193 | 0.73392 | 0.05559 | 2429.96933 | 20.77252 |
| 010116 | Amazonas | Olleros | menor error | 0.005198 | 0.09114 | 0.03463 | 0.04355 | 0.0 | 3163.64666 | 19.08744 |
| 220509 | San Martin | Shanao | mayor error | 0.023815 | 0.49916 | 0.0 | 1.52697 | 0.12304 | 501.27412 | 12.5991 |
| 220703 | San Martin | Caspisapa | mayor error | 0.021004 | 0.47155 | 0.0 | 0.56468 | 0.03059 | 343.67737 | 9.43027 |
| 060905 | Cajamarca | Namballe | mayor error | 0.020564 | 0.32393 | 0.54108 | 0.26055 | 0.00311 | 1988.52372 | 25.16711 |


### 4.4 Recomendación — factores principales

| factor | max_abs_r | min_p | diff_estandarizada | rank_correlacion | rank_significancia | rank_grupos | score_combinado | recomendado_top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elev_media_m | 0.4962 | 0.0261 | 0.9256 | 6.0 | 6.0 | 6.0 | 6.0 | True |
| pct_anp | 0.3911 | 0.0881 | 0.7042 | 5.0 | 5.0 | 5.0 | 5.0 | True |
| pct_agropecuario | 0.3203 | 0.1686 | 0.3058 | 3.0 | 3.0 | 3.0 | 3.0 | True |
| densidad_carreteras_km_km2 | 0.3636 | 0.1151 | 0.1154 | 4.0 | 4.0 | 1.0 | 3.0 | False |
| densidad_rios_km_km2 | 0.2393 | 0.3097 | 0.3467 | 2.0 | 2.0 | 4.0 | 2.67 | False |
| pendiente_media_deg | 0.1669 | 0.4818 | 0.1981 | 1.0 | 1.0 | 2.0 | 1.33 | False |


## 5. Cajamarca y departamentos amazónicos limítrofes

- **Cajamarca** (2 distrito(s)): Huabal (RMSE=0.0071, error medio); Namballe (RMSE=0.0206, mayor error).
- **Amazonas** (4 distrito(s)): Mariscal Castilla (RMSE=0.0028, menor error); Chachapoyas (RMSE=0.0036, menor error); Olleros (RMSE=0.0052, menor error); Inguilpata (RMSE=0.0117, mayor error).
- **San Martin** (9 distrito(s)): Habana (RMSE=0.0066, menor error); Cacatachi (RMSE=0.0070, menor error); Jepelacio (RMSE=0.0072, error medio); Soritor (RMSE=0.0079, error medio); Tingo De Saposoa (RMSE=0.0085, error medio); Rioja (RMSE=0.0087, mayor error); Tingo De Ponasa (RMSE=0.0144, mayor error); Caspisapa (RMSE=0.0210, mayor error); Shanao (RMSE=0.0238, mayor error).

## Lectura

- **CNN**: el RMSE en las 100 predicciones de generalización (20 zonas × 5 años) empeora un 3.5% respecto al RMSE en muestra (0.010912 → 0.011290).
- `elev_media_m` se correlaciona significativamente con el error por zona (r_pearson=-0.3959 p=0.084, r_spearman=-0.4962 p=0.0261); media en grupo de mayor error=944.49173 vs. menor error=2833.03473.
- **Factores territoriales principales** (score combinado, top 3): `elev_media_m`, `pct_anp`, `pct_agropecuario` — son los que muestran mayor evidencia conjunta de correlación, significancia y separación entre los grupos de menor y mayor error.