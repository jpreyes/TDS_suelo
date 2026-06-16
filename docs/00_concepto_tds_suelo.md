# Concepto TDS-Suelo

## Definicion

TDS-Suelo significa, para este proyecto, dinamica estructural transicional aplicada al suelo y a la propagacion sismica.

El sistema fisico no se modela como una simple relacion GMPE:

```text
Mw + distancia + sitio -> intensidad
```

sino como una transferencia multiescala:

```text
fuente 3D -> ruta/grafo -> receptor/suelo -> respuesta observada
```

## Diferencia Con GMPE

Un GMPE busca predecir medidas de movimiento fuerte.

TDS-Suelo busca primero entender la estructura que causa esas medidas:

```text
que parte viene de la fuente
que parte viene de la ruta
que parte viene del suelo
que parte viene de una falla, cuenca o ruptura
que parte cambia con la direccion de llegada
```

Las salidas PGA/PSA/Arias son productos derivados. El producto principal inicial es un atlas de transferencia y anomalias.

## Pregunta Fisica Principal

Para un mismo receptor:

```text
por que un sismo desde el norte puede activar un pico cerca de 8 Hz,
uno desde el sur cerca de 12 Hz,
y uno frontal cerca de 10 Hz?
```

La respuesta no debe buscarse solo en el receptor. Debe buscarse en:

```text
fuente 3D
ruta
backazimut
incidencia
suelo
geologia local y regional
```

## Rol De Kozyrev

Kozyrev entra como herramienta multiescala para detectar:

```text
saltos
bordes
cambios de regimen
fallas
cuencas
rutas anomalas
anisotropia direccional
```

Los coeficientes Kozyrev no son decorativos. Deben interpretarse como posible evidencia de cambios de transferencia.

## Regla De Trabajo

No se inventa geologia en hojas sin datos.

Primero:

```text
records reales -> targets fisicos -> residuos -> modos latentes -> grafo/Kozyrev -> atlas
```

Despues:

```text
atlas validado -> forward condicionado -> acelerogramas opcionales
```

