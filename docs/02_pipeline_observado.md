# Pipeline Observado

## Ejecutar

```bash
tsd-suelo build --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs
```

Para una prueba rapida:

```bash
tsd-suelo build --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs/smoke_real --max-h5 10
```

Para precomputar todos los H5 y todos los parquets reutilizables:

```bash
time tsd-suelo build \
  --records-dir ../records \
  --flatfiles-dir ../flatfiles \
  --output-dir outputs_precomputed \
  --workers 8 \
  --progress-every 500
```

Para reutilizar todo despues:

```bash
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --reuse-products --workers 8
```

El progreso queda en pantalla y en:

```text
outputs_precomputed/run.log
outputs_precomputed/_h5_target_batches/
```

Monitoreo:

```bash
tail -f outputs_precomputed/run.log
```

Cada fase registra inicio, fin, duracion, filas generadas y avance H5 con porcentaje, velocidad y ETA.

Si la corrida se interrumpe durante H5, repetir el mismo comando reutiliza los batches en `_h5_target_batches` y procesa solo lo pendiente.

Si la corrida se corta despues de `waveform_targets_observed.parquet`, pero antes de terminar todos los productos:

```bash
tsd-suelo build --records-dir ../records --flatfiles-dir ../flatfiles --output-dir outputs_precomputed --reuse-targets --workers 8
```

## Convenciones De Claves

La clave canonica `event_id` sale del nombre del H5:

```text
YYYYMMDDHHMMSS_STATION.h5
```

El identificador interno `EventID_BM16` del H5 se conserva como `h5_event_id_bm16`, pero no se usa para unir contra flatfiles porque no siempre coincide con `EventID` de los CSV.

## H5 Y Flatfile Completo

Cuando hay H5 para un `event_id + station_id`, los targets calculados desde H5 tienen prioridad. Cuando no hay H5, el sistema incorpora el registro del flatfile como `observed_source = flatfile` usando targets observados publicados:

```text
PGA N/E/Z/H
Arias N/E/Z/H cuando existe
duraciones 5-75 y 5-95 cuando existen
PSA RotD50 en 0.1, 0.2, 0.5, 1.0 y 2.0 s
```

Esto permite usar los mas de 40 mil registros del flatfile sin inventar formas de onda.

## Mascara De Chile

Por defecto se aplica una mascara gruesa incorporada y se escribe:

```text
outputs/chile_mask.geojson
```

Para una mascara oficial:

```bash
tsd-suelo build --mask-geojson ../geodata/chile_mask.geojson --records-dir ../records --flatfiles-dir ../records/flatfiles --output-dir outputs
```

## Targets H5

Los H5 se interpretan como aceleracion en `cm/s2` y se exportan tambien en `g`.

Targets principales:

```text
PGA E/N/Z/H
Arias E/N/Z/H
CAV E/N/Z/H
duracion 5-75 y 5-95
energia espectral por bandas
frecuencia dominante
centroide y ancho espectral
PSA 5% en T = 0.1, 0.2, 0.5, 1.0, 2.0 s
polarizacion horizontal
razones H/V y E/N
```

## Salida Para Atlas

`atlas_geologico.geojson` contiene features de:

```text
receptores
fuentes
rutas
campos Kozyrev de mayor delta
grilla ultrametrica probabilistica
candidatos de falla observados
```

`atlas_geologico.kmz` contiene la misma informacion en formato KML comprimido para inspeccion rapida.

## Reporte

El build genera:

```text
results_report.html
results_summary.json
top_kozyrev_anomalies.csv
top_receiver_anomalies.csv
top_route_anomalies.csv
top_fault_candidates.csv
```

## Grafo Ultrametrico Kozyrev

El build escribe la grilla Kozyrev completa como grafo:

```text
kozyrev_ultrametric_nodes.parquet
kozyrev_ultrametric_edges.parquet
kozyrev_ultrametric_nodes.geojson
kozyrev_ultrametric_edges.geojson
kozyrev_heatmap.geojson
kozyrev_heatmap.kmz
```

Cada celda `source3d`, `route` y `receiver` en los niveles `j1..j4` es un nodo. Cada relacion padre-hijo entre niveles es una arista `ultrametric_parent_child`. Las relaciones fuente-ruta-receptor tambien quedan en la misma tabla como `source_route_receiver`.

Campos de lectura directa:

```text
failure_probability_pct
edge_probability_pct
probability_basis
```

Los porcentajes son probabilidad empirica relativa observada, no probabilidad absoluta calibrada contra un catalogo oficial de fallas. Para el mapa de calor completo, abrir `kozyrev_heatmap.geojson` o `kozyrev_heatmap.kmz`.

## Fallas Candidatas

El pipeline identifica lineamientos candidatos, no fallas oficiales nombradas. La capa se escribe en:

```text
fault_candidates.parquet
fault_candidates.geojson
fault_candidates.kmz
```

Cada candidato representa un corredor fuente-ruta-receptor con concentracion de modos residuales, salto Kozyrev y repeticion de registros. Para interpretacion geologica final, cruzar esta capa con trazas oficiales de fallas y mapas geologicos en QGIS.

## Dinamica Compatible Forward-Ready

El producto forward-ready se escribe en:

```text
compatible_dynamics.parquet
forward_conditioning_profiles.parquet
forward_conditioning_template.json
```

`compatible_dynamics.parquet` conserva la dinamica observada por registro:

```text
baseline_known_site_log_<target>
dynamic_correction_log_<target>
compatible_log_<target>
compatible_<target>
dynamic_anomaly_score
forward_support_weight
```

`forward_conditioning_profiles.parquet` resume correcciones por contexto multiescala:

```text
source3d
route
receiver
fault_candidate
```

La ecuacion minima para un forward posterior queda:

```text
log(target_forward) = baseline_source_distance_site_log + dynamic_correction_log(context)
```

La tabla no genera sismogramas sinteticos. Es el condicionamiento observado que debe alimentar un forward posterior validado fuera de muestra.

Para ver por SSH:

```bash
tsd-suelo summary --output-dir outputs --top-n 20
python -m http.server 8000 -d outputs
```

Luego abrir con tunel local:

```bash
ssh -L 8000:localhost:8000 usuario@servidor
```
