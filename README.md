# TSD-Suelo

TSD-Suelo es un sistema independiente para identificar dinamica compatible del suelo desde observaciones. No es un GMPE clasico y no hereda scripts, parquets, runners ni dependencias internas de GMPE, Modelo E o TSD estructural.

Fuentes primarias permitidas, normalmente fuera de la carpeta del proyecto:

```text
../records/*.h5
../records/flatfiles/*.csv
```

## Objetivo

Descubrir fallas, rupturas, cuencas, rutas anomalas, anisotropia direccional y comportamiento de suelo mediante una lectura multiescala fuente-ruta-receptor:

```text
fuente 3D -> ruta -> receptor/suelo -> targets fisicos observados -> residuos -> modos latentes -> grafo Kozyrev -> atlas
```

El forward condicionado queda preparado como contrato posterior, pero no se ajusta ni genera acelerogramas sinteticos en esta etapa.

## Instalacion En Linux Mint

```bash
git clone <URL_DEL_REPO> TDS_suelo
cd TDS_suelo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Por defecto el proyecto busca datos en rutas relativas:

```text
../records
../records/flatfiles
```

Tambien puedes pasar rutas relativas explicitas fuera del repo:

```bash
tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../records/flatfiles \
  --output-dir outputs
```

## CLI

```bash
tsd-suelo inventory
tsd-suelo targets --max-h5 20
tsd-suelo build --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs
tsd-suelo forward --output-dir outputs
tsd-suelo summary --output-dir outputs
tsd-suelo report --output-dir outputs
```

`build` usa todos los H5 disponibles y agrega registros `flatfile-only` cuando no hay H5 correspondiente. Para usar solo H5:

```bash
tsd-suelo build --h5-only --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs_h5
```

## Corrida Grande Con Muchos H5

Para dejar todo precomputado una vez, incluyendo todos los H5 y PSA, usa una carpeta de salida estable. Puede demorar muchas horas; queda todo en parquets reutilizables:

```bash
time tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_precomputed \
  --workers 8 \
  --progress-every 500
```

Despues de esa corrida, reutiliza todos los productos sin volver a leer H5 ni recalcular ETL/residuos/modos/grafo:

```bash
tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_precomputed \
  --reuse-products \
  --workers 8 \
  --progress-every 500
```

Puedes elegir el modo de analisis:

```bash
# Solo grilla espacial de anomalias/fallas
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --analysis-mode spatial --reuse-products

# Solo red dinamica espectral equivalente
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --analysis-mode spectral --reuse-products

# Ambos modos
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --analysis-mode both --reuse-products
```

El modo `spectral` reabre los H5 la primera vez para construir firmas espectrales completas en una grilla de frecuencias comun; despues queda reutilizable como parquet.

Para recalcular solo la dinamica compatible y los perfiles de forward desde productos observados ya existentes, sin releer H5:

```bash
tsd-suelo forward \
  --output-dir outputs_precomputed \
  --top-n 80
```

Ese comando actualiza:

```text
compatible_dynamics.parquet
forward_conditioning_profiles.parquet
forward_conditioning_template.json
forward_manifest.json
results_report.html
results_summary.json
```

Si se corta despues de haber calculado `waveform_targets_observed.parquet`, pero antes de terminar todo, puedes retomar sin releer H5:

```bash
tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_precomputed \
  --reuse-targets \
  --workers 8 \
  --progress-every 500
```

Para decenas de miles de H5, si quieres una primera version rapida antes de la corrida completa, agrega `--skip-psa`. Esa version no reemplaza la corrida final con PSA.

Para medir tiempo antes del build completo:

```bash
time tsd-suelo targets \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_bench \
  --max-h5 200 \
  --workers 8 \
  --skip-psa
```

El archivo `outputs_bench/waveform_targets_observed.meta.json` deja registrado `h5_processed`, `workers` y si se calculo PSA.

El build escribe progreso en pantalla y tambien en:

```text
outputs_precomputed/run.log
outputs_precomputed/_h5_target_batches/
```

Para monitorear desde otra sesion SSH/tmux:

```bash
tail -f outputs_precomputed/run.log
```

Si la corrida se cae durante la lectura de H5, repite el mismo comando. El extractor lee los batches existentes en `_h5_target_batches` y continua con los H5 pendientes. No necesitas esperar a que exista el parquet final para retomar.

Para escribir solo al log, sin imprimir en pantalla:

```bash
tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_precomputed \
  --workers 8 \
  --quiet
```

Puedes cambiar el archivo de log:

```bash
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --log-file logs/tsd_suelo.log
```

## Mascara De Chile

El build aplica una mascara gruesa incorporada de Chile y escribe `outputs/chile_mask.geojson`. Para usar una mascara oficial local:

```bash
tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../records/flatfiles \
  --mask-geojson ../geodata/chile_mask.geojson \
  --output-dir outputs
```

Para diagnostico sin mascara:

```bash
tsd-suelo build --no-chile-mask --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs_nomask
```

## Productos

El build completo escribe productos derivados en `outputs/`:

```text
observed_inventory.json
waveform_targets_observed.parquet
record_geometry.parquet
receiver_index.parquet
source3d_index.parquet
geo_targets_observed.parquet
geo_residuals.parquet
target_level_attribution.csv
latent_modes.parquet
latent_mode_components.csv
spatial_grid_nodes.parquet
spatial_grid_edges.parquet
spatial_anomaly_nodes.geojson
spatial_fault_edges.geojson
spatial_probability_heatmap.geojson
spatial_probability_heatmap.kmz
spectral_record_signatures.parquet
spectral_node_dynamics.parquet
spectral_edge_transmissibility.parquet
spectral_dynamic_modes.parquet
spectral_mode_components.csv
spectral_dynamic_heatmap.geojson
spectral_dynamic_heatmap.kmz
spectral_frequency_grid.json
route_graph_observed.parquet
kozyrev_graph_fields.parquet
kozyrev_ultrametric_nodes.parquet
kozyrev_ultrametric_edges.parquet
kozyrev_ultrametric_nodes.geojson
kozyrev_ultrametric_edges.geojson
kozyrev_heatmap.geojson
kozyrev_heatmap.kmz
fault_candidates.parquet
top_fault_candidates.csv
fault_candidates.geojson
fault_candidates.kmz
compatible_dynamics.parquet
forward_conditioning_profiles.parquet
atlas_geologico.geojson
atlas_geologico.kmz
chile_mask.geojson
forward_conditioning_template.json
pipeline_manifest.json
results_report.html
results_summary.json
top_kozyrev_anomalies.csv
top_receiver_anomalies.csv
top_route_anomalies.csv
```

`outputs/` esta ignorado por git porque son artefactos reproducibles.

## Identificar Anomalias Y Fallas Candidatas

La corrida no asigna nombres oficiales de fallas. Produce una grilla espacial jerarquica rectangular y lineamientos candidatos observados desde los registros, despues de residualizar por fuente/distancia/sitio conocido. Revisa primero:

```bash
tsd-suelo summary --output-dir outputs_precomputed --top-n 20
```

Productos principales:

```text
spatial_grid_nodes.parquet
spatial_grid_edges.parquet
spatial_probability_heatmap.geojson
spatial_probability_heatmap.kmz
spatial_anomaly_nodes.geojson
spatial_fault_edges.geojson
kozyrev_ultrametric_nodes.parquet
kozyrev_ultrametric_edges.parquet
kozyrev_heatmap.geojson
kozyrev_heatmap.kmz
fault_candidates.parquet
fault_candidates.geojson
fault_candidates.kmz
```

En la grilla espacial, cada celda ocupada en niveles `J1..J12` queda como nodo centrado. Las aristas unen vecinos rectangulares `N/S/E/O` y diagonales. Una falla candidata local aparece como salto dinamico entre celdas vecinas.

Los campos principales son:

```text
anomaly_probability_pct
fault_probability_pct
failure_probability_pct
edge_probability_pct
probability_basis
```

Los porcentajes son probabilidades empiricas relativas observadas, calculadas por nivel desde percentiles de norma modal, intensidad, soporte y salto entre vecinos. No son probabilidades absolutas calibradas con fallas catalogadas. Abre `spatial_probability_heatmap.geojson` o `spatial_probability_heatmap.kmz` en QGIS/Google Earth para ver el mapa de calor espacial. Cruza esas capas con cartografia de fallas oficial si necesitas nombres geologicos.

## Red Dinamica Espectral Equivalente

El modo `spectral` implementa la analogia de estructura equivalente: cada celda espacial ocupada es un nodo/sensor, las celdas vecinas son aristas de transferencia y cada registro H5 aporta una firma espectral horizontal sobre una grilla comun de frecuencias. La estimacion usa todas las frecuencias simultaneamente, no bandas independientes.

Productos:

```text
spectral_record_signatures.parquet
spectral_node_dynamics.parquet
spectral_edge_transmissibility.parquet
spectral_dynamic_modes.parquet
spectral_mode_components.csv
spectral_dynamic_heatmap.geojson
spectral_dynamic_heatmap.kmz
spectral_frequency_grid.json
```

Campos principales:

```text
spectral_dynamic_probability_pct
spectral_transfer_probability_pct
spectral_jump_norm
transfer_log_f000 ... transfer_log_f063
```

`spectral_edge_transmissibility.parquet` es el operador inicial para forward espectral: cada arista contiene una diferencia log-espectral entre nodos vecinos sobre toda la grilla de frecuencias. No es FEM/BEM; es una red dinamica equivalente calibrada directamente desde los H5.

## Dinamica Compatible Para Forward

El build genera una capa condicionante para forward posterior:

```text
compatible_dynamics.parquet
forward_conditioning_profiles.parquet
forward_conditioning_template.json
```

`compatible_dynamics.parquet` tiene una fila por registro observado con geometria, sitio conocido, modos latentes, campos Kozyrev, falla candidata asociada, targets observados, baseline por fuente/distancia/sitio y correcciones dinamicas:

```text
baseline_known_site_log_<target>
dynamic_correction_log_<target>
compatible_log_<target>
compatible_<target>
dynamic_anomaly_score
forward_support_weight
```

La forma operativa para un forward condicionado es:

```text
log(target_forward) = baseline_source_distance_site_log + dynamic_correction_log(context)
```

`forward_conditioning_profiles.parquet` agrega esas correcciones por contexto `source3d`, `route`, `receiver` y `fault_candidate`, para reutilizarlas sobre geometria nueva o escenarios cercanos. Antes de usarlo como predictor, valida fuera de muestra.

## Flujo Implementado

1. Inventario observado de H5, eventos, records y estaciones.
2. ETL desde H5 y flatfiles.
3. Geometria fuente-receptor: distancia, azimut, backazimut, incidencia y celdas multiescala.
4. Targets fisicos desde H5 cuando existe forma de onda.
5. Targets observados del flatfile para registros sin H5.
6. Tabla `geo_targets_observed.parquet`.
7. Residualizacion por fuente/evento, magnitud, profundidad, distancia y sitio conocido (`Vs30`, HVSR, `kappa0`, pendiente).
8. Modos latentes por PCA sobre residuos fisicos.
9. Proyeccion en grafo Kozyrev fuente 3D -> ruta -> receptor.
10. Atlas geologico GeoJSON/KMZ y reporte HTML.
11. Plantilla de forward condicionado posterior.

## Tests

```bash
pytest
```

Las pruebas usan H5 y flatfiles sinteticos temporales; no dependen de los datos reales.

## Ver Resultados Por SSH

Resumen en terminal:

```bash
tsd-suelo summary --output-dir outputs_precomputed --top-n 20
```

Reporte HTML:

```bash
tsd-suelo serve \
  --output-dir outputs_precomputed \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --host 127.0.0.1 \
  --port 8787
```

Para habilitar ejecucion desde la pagina admin:

```bash
export TSD_SUELO_ADMIN_TOKEN='cambia-este-token-largo'
tsd-suelo serve \
  --output-dir outputs_precomputed \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --host 127.0.0.1 \
  --port 8787
```

Luego abre `https://tsd.jpreyes.cl/admin`. Desde esa pagina puedes lanzar `build`, ejecutar `forward` condicionado desde parquets existentes, hacer `git pull --ff-only`, `pip install -e .`, detener un proceso, ver logs y descargar productos. Usa esto solo con token fuerte y preferentemente detras de Cloudflare Access.

Abre `https://tsd.jpreyes.cl/results_report.html`. El reporte incluye un mapa interactivo con OpenStreetMap/Leaflet, capas de Chile, dinamica espectral, fallas candidatas, grilla espacial, Kozyrev y atlas.

Tambien puedes descargar `outputs/atlas_geologico.geojson` o `outputs/atlas_geologico.kmz` y abrirlos en QGIS/Google Earth.
