# Proyecto AHPC — Ecuación de Calor (Heat Equation)
## Informe de seguimiento: avance y estatus para la Entrega Parcial

> **Repositorio:** https://github.com/hugoangeles0810/ahpc-heat-equation
> **Curso:** CS8084 — Applied High Performance Computing, Posgrado UTEC 2026-I
> **Profesor:** José Fiestas
> **Proyecto elegido:** Opción 4 — *Ecuación de calor* (análisis de performance de un solver MPI que resuelve la EDP de calor en paralelo)
> **Última revisión analizada:** commit `c7adab8` (2026-06-10)
> **Informe parcial:** ✅ `Informe_Parcial.tex` → `Informe_Parcial.pdf` (20 páginas) — completado
> **Fechas clave:** Entrega parcial → **domingo 14/06/2026** · Entrega final → **viernes 17/07/2026** · Sustentación → **sábado 18/07/2026**

---

## 0. Resumen ejecutivo

El repositorio del grupo está **bastante avanzado para ser una entrega parcial**: ya existe
código funcional (versión serial y versión MPI con descomposición de dominio 2D + halos),
un documento de **PRAM y complejidad teórica muy completo** (10 páginas, con DAG,
análisis trabajo–profundidad, isoeficiencia y número óptimo de procesos), infraestructura
de benchmarking para el clúster **Khipu** (SLURM), dos corridas experimentales reales
(`exp01`, `exp02`) y un script de análisis que genera las gráficas de tiempo, *speedup*,
eficiencia e iteraciones.

El equipo ya **completó el empaquetado que pide la sección 7.1 del enunciado**: existe el
**informe parcial consolidado** (`Informe_Parcial.tex` → `Informe_Parcial.pdf`, 20
páginas) con nombre del proyecto e integrantes, **% de participación** (100% para los 4
integrantes), introducción, método (PRAM + métricas), resultados preliminares,
conclusiones y bibliografía con declaración de uso de IA.

El único punto que queda abierto es metodológico, no documental: el criterio de
convergencia (`itmax=20000`) **no se alcanza** en ninguna corrida, por lo que la
validación teórica de `T = Θ(n²)` (sección 4 de `pram-y-complejidad.md`) todavía no está
confirmada empíricamente. Esto ya quedó **documentado como limitación conocida** dentro
del informe parcial y se deja como trabajo pendiente para la entrega final. Ver detalle
en §7.

**Estado global estimado: ~100% del contenido y del empaquetado de la entrega parcial
está listo.** Solo falta subir el PDF a Canvas (entrega administrativa).

---

## 1. Datos del proyecto

| Campo | Valor |
|---|---|
| Nombre del proyecto | Análisis de performance del solver MPI de la Ecuación de Calor (2D, diferencias finitas) |
| Tema del enunciado | Sección 4 — *Ecuación de calor* (PDF `HPC-Proyecto-2026I-1.pdf`) |
| Repositorio | [`hugoangeles0810/ahpc-heat-equation`](https://github.com/hugoangeles0810/ahpc-heat-equation) |
| Integrantes (según `README.md` del repo) | Hugo Angeles, Christian Cajusol, Jhomar Yurivilca, Francisco Meza |
| % de participación | ✅ **100%** para los 4 integrantes (Hugo Angeles, Christian Cajusol, Jhomar Yurivilca, Francisco Meza) — Tabla "Participación del equipo" de `Informe_Parcial.tex` |
| Clúster usado | Khipu (UTEC), cuenta `postgrado`, nodos Xeon Gold 6130 |
| Commits totales | 27 (todos sobre `main`, primer commit 2026-06-07, último 2026-06-10) |

> **Nota sobre commits:** los 27 commits del repo están firmados por *Hugo Angeles*. Esto
> no necesariamente refleja el reparto real de trabajo (puede haber diseño, redacción del
> PRAM, revisión, ejecución de pruebas en Khipu, etc. hechas por otros integrantes fuera
> de Git), pero **es un dato a tener en cuenta al definir el % de participación** que pide
> la sección 7.1 del enunciado — conviene que cada integrante registre su aporte
> (commits, secciones del documento PRAM, corridas en Khipu, redacción del informe, etc.).

---

## 2. Qué exige la entrega parcial (sección 7.1 / 7.2 del enunciado)

El informe parcial debe incluir:

1. **Nombre del proyecto y de los integrantes**
2. **Porcentaje de participación** de cada integrante (0–100%)
3. **Introducción**: proyecto elegido y solución planteada
4. **Método** (rúbrica punto 1): algoritmo usado (PRAM) + métricas solicitadas
   (tiempo de ejecución, *speedup*, eficiencia)
5. **Bibliografía** (rúbrica punto 3): fuentes usadas (incl. IA) y su impacto

La rúbrica de la entrega parcial pondera:

| # | Criterio | Puntos | Estado actual |
|---|---|---|---|
| 1 | Desarrollo de PRAM y métricas | 6 | ✅ **Logrado** — `docs/pram-y-complejidad.md` cubre PRAM, DAG, work-depth, Brent, isoeficiencia y las 3 métricas pedidas |
| 2 | Uso correcto de fuentes externas | 4 | ✅ **Logrado** — 10 referencias citadas con su rol específico + nota explícita de uso de IA (Claude) |
| 3 | Presentación escrita (informe) y trabajo en grupo | 4 | ✅ **Logrado** — informe consolidado `Informe_Parcial.tex`/`.pdf` (20 páginas) con nombre, integrantes, % de participación, introducción, método y bibliografía |
| 4 | Presentación oral | 6 | ⏳ N/A todavía — corresponde a la sustentación/avance oral, no al repo |

---

## 3. Avance técnico detallado

### 3.1 Código del solver (`src/`)

- **`heat-big.c`** (168 líneas): versión **serial** de referencia. Resuelve la ecuación de
  calor 2D con diferencias finitas (stencil de 5 puntos), esquema explícito de Jacobi
  (Euler hacia adelante), `dt = min(dx²,dy²)/4` (límite CFL) y criterio de parada
  `dphimax < eps` (`eps = 1e-8`).
- **`heat-mpi-big.c`** (350 líneas): versión **paralela MPI**. Implementa:
  - Descomposición de dominio 2D vía `MPI_Dims_create` + comunicador cartesiano
    (`MPI_Cart_create`, `MPI_Cart_coords`, `MPI_Cart_shift`).
  - Reparto de la malla entre procesos (incluyendo el caso no divisible exacto).
  - Intercambio de **halos** (ghost cells) con `MPI_Irecv`/`MPI_Isend`/`MPI_Waitall` y
    tipos derivados (`MPI_Type_create_subarray`) para filas/columnas no contiguas.
  - **`MPI_Allreduce`** del criterio de convergencia cada `stride = 10` iteraciones
    (optimización justificada en el documento PRAM, sección 3).
  - Cronometraje separado de **cómputo total**, **comunicación de halos** (`comm_time`)
    y **criterio de parada** (`criterion_time`) vía `MPI_Wtime`.
  - Ambos binarios son **parametrizables** por línea de comandos:
    `bin/heat-mpi-big <grid_size> <itmax> <print_result>`.
  - Imprimen una línea `RESULT,...` machine-readable con
    `grid_size,nprocs,idim,kdim,iterations,wall_time_s,comm_time_s,criterion_time_s`,
    consumida luego por `scripts/collect_results.sh`.

✅ **Estado:** funcional, parametrizado, "endurecido" (refactors `5c6aacd`/`785226f`),
compila limpio (`-Wall -Wextra` sin warnings tras `eae21a4`).

### 3.2 PRAM y complejidad teórica (`docs/pram-y-complejidad.md`)

Documento de ~400 líneas, muy completo, que cubre **objetivo (a)** del enunciado punto
por punto:

- Justifica que el solver es **CREW** (lectura concurrente del stencil de 5 puntos,
  escritura exclusiva por doble buffer Jacobi).
- Traduce el bucle real de `heat-mpi-big.c` (líneas 239–297) a notación PRAM
  (`forall` + reducción).
- DAG por iteración (Mermaid) con anotación de **trabajo** y **profundidad** por fase.
- Deriva **T = Θ(n²)** pasos hasta convergencia a partir de la condición CFL y la tasa de
  convergencia de Jacobi sobre Laplace, con una validación empírica citada (malla 80×80 →
  ~14 300 iteraciones).
- Análisis **trabajo–profundidad**: `W₁ = Θ(n⁴)`, `D∞ = Θ(n² log n)`, paralelismo
  disponible `Θ(n²/log n)`, cota de Brent.
- **Tp, Speedup y Eficiencia** en el modelo PRAM puro:
  `Sp = p / (1 + p·log p / n²)`, `Ep = 1 / (1 + p·log p / n²)`.
- Extiende el modelo a **memoria distribuida** (BSP, costo `tₛ + tw·m`): costo de halos
  `Θ(n/√p)`, costo de la reducción `Θ(tₛ log p / stride)`, razón
  comunicación/cómputo `√p / n`.
- **Isoeficiencia** `Θ(p² log² p)` y **número óptimo de procesos**
  `p_opt = Θ(n²)` (acotado en la práctica por la constante `(c_f/t_w)²`).
- Sección final que **explica cómo normalizar** la fórmula teórica contra el CSV de
  mediciones (mínimos cuadrados sobre `a·n⁴/p + b·n³/√p + c`) — pero **esto todavía no
  está implementado en `analysis/plot_performance.py`** (ver §7).

✅ **Estado:** objetivo (a) del enunciado, **muy bien cubierto** — probablemente el punto
más fuerte del proyecto en este momento.

### 3.3 Infraestructura de benchmarking en Khipu (`scripts/`)

- **`Makefile`**: `make` compila `bin/heat-big` (serial, `cc`) y `bin/heat-mpi-big`
  (`mpicc`), ambos con `-O3 -Wall -Wextra`.
- **`scripts/run_local.sh`**: smoke test rápido con `mpirun --oversubscribe` para laptop.
- **`scripts/benchmark.slurm`**: job SLURM que barre **2 ejes**:
  - tamaños de malla `NS = (128, 256, 512, 1024)`
  - número de procesos `PROCS = (1, 2, 4, 8, 16)`
  con `ITMAX = 20000`, `gnu12/12.4.0` + `openmpi4/4.1.6`. Cada corrida se vuelca a
  `results/raw/run_n<N>_p<P>.out` y se consolida en `results/benchmark.csv`.
- **`scripts/collect_results.sh`**: extrae la línea `RESULT,...` de cada salida y la
  convierte en una fila CSV.
- **Restricción real de la cuenta `postgrado`** (documentada y resuelta tras varias
  iteraciones, commits `8e41f02`→`b9f4f17`→`478dca6`): tope de `cpu=32` CPUs lógicas por
  usuario, nodos con *hyperthreading* (32 núcleos físicos = 64 CPUs lógicas). Por eso el
  job usa `--ntasks=16 --hint=nomultithread` (16 núcleos físicos, sin `--exclusive`) en
  vez de pedir el nodo completo.

✅ **Estado:** infraestructura de benchmarking **funcionando de punta a punta sobre
Khipu real** (no solo localmente) — esto es un avance significativo, ya que requiere
acceso al clúster, módulos correctos y ajuste de la configuración SLURM a los límites
reales de la cuenta.

### 3.4 Experimentos ejecutados (`results/`)

Dos experimentos completos, documentados en
[`results/EXPERIMENTS.md`](https://github.com/hugoangeles0810/ahpc-heat-equation/blob/main/results/EXPERIMENTS.md):

| Exp | Configuración | Nodos × cores | Hallazgo principal |
|---|---|---|---|
| **exp01** | `--nodelist=n003 --nodes=1 --ntasks=16` | 1 × 16 (n003) | Escalado fuerte en 1 nodo. `n=128` satura en `S≈4` desde `p≥8` (comunicación domina); `n=1024` casi lineal (`S=13.4` en `p=16`). |
| **exp02** | `--nodelist=n003,n005 --nodes=2 --ntasks-per-node=8` | 2 × 8 (n003+n005) | `P=16` repartido 8+8 entre dos nodos **supera** al caso de 1 nodo. |

Cada experimento incluye `benchmark.csv` + 4 figuras (`time.png`, `speedup.png`,
`efficiency.png`, `iterations.png`).

✅ **Estado:** objetivo (b) del enunciado — *"ejecute mediciones de tiempo... para
distinto número de procesos y tamaño del problema"* — **cubierto en cuanto a barrido
p×n**. La parte de *"compárelas con la teórica"* todavía es parcial (ver §7).

### 3.5 Análisis y gráficas (`analysis/plot_performance.py`)

Script Python (pandas + matplotlib) que, a partir de `results/benchmark.csv`, calcula:

- **Tiempo de ejecución** vs. `p` (log-log, una curva por tamaño de malla `n`)
- **Speedup** `S(p) = T(1)/T(p)` vs. `p`, con referencia lineal ideal
- **Eficiencia** `E(p) = S(p)/p` vs. `p`, con referencia ideal = 1.0
- **Iteraciones hasta converger** vs. `n`, con referencia `Θ(n²)`

✅ **Estado:** objetivo (c) del enunciado — *"desarrolle un software de análisis de
performance... que genere la representación gráfica"* — **implementado y produce las 4
gráficas pedidas**. Pendiente: las curvas "ideales" actuales son el caso PRAM sin
comunicación (lineal / `E=1`); falta superponer la **curva teórica con comunicación**
derivada en `pram-y-complejidad.md` §9 (`Ep = 1/(1+Θ(√p/n))`) para una validación más
fina.

### 3.6 Documentación didáctica (`docs/heat-equation.md` + `docs/generate_figures.py`)

Documento de 521 líneas que explica, paso a paso y con figuras generadas (animación de
convergencia, etapas de difusión, mapa inicial vs. estado estacionario):

- La física de la ecuación de calor (1D → 2D → Laplace en estado estacionario)
- Diferencias finitas y el stencil de 5 puntos
- Esquema explícito, estabilidad CFL y criterio de convergencia
- Mapeo línea por línea al código C
- Descomposición de dominio 2D, halos y comunicación MPI

✅ **Estado:** no es un requisito explícito de la rúbrica, pero **agrega mucho valor**
para la sección de Introducción/Método del informe y para la presentación oral (dominio
del tema).

---

## 4. Resultados obtenidos hasta el momento

### Tiempo de pared (segundos), `p=1` vs `p=16`, exp01 (1 nodo)

| n (malla) | T(p=1) | T(p=16) | Speedup S(16) | Eficiencia E(16) |
|---|---|---|---|---|
| 128  | 0.633 s | 0.158 s | 4.0  | 0.25 |
| 256  | 2.598 s | 0.329 s | 7.9  | 0.49 |
| 512  | 12.056 s | 0.973 s | 12.4 | 0.77 |
| 1024 | 46.676 s | 3.472 s | 13.4 | 0.84 |

### Comparación 1 nodo (exp01) vs 2 nodos (exp02), `P=16`

| n | exp01 (1 nodo) tiempo / efic. / comm | exp02 (2 nodos) tiempo / efic. / comm |
|---|---|---|
| 512  | 0.973 s / 0.77 / 0.205 s | **0.890 s / 0.84 / 0.195 s** |
| 1024 | 3.472 s / 0.84 / 0.459 s | **3.205 s / 0.94 / 0.351 s** |

**Lectura de los resultados:**

1. **A mayor `n`, mejor escalabilidad** — consistente con la predicción teórica: la
   razón comunicación/cómputo es `√p/n`, así que mallas grandes amortizan mejor el costo
   de los halos. `n=128` se queda en `S≈4` (saturado), `n=1024` llega a `S≈13.4` de 16.
2. **Repartir entre 2 nodos es más rápido que concentrar en 1** (a igual `P=16`): el
   solver está limitado por **ancho de banda de memoria**, no por latencia de red — dos
   nodos aportan más ancho de banda agregado del que cuesta cruzar la red. Es un
   resultado *contraintuitivo* y un buen punto de discusión para el informe.
3. **El tiempo de comunicación crece con `p`** en términos absolutos (de ~0.0005 s a
   ~0.1–0.46 s), pero se mantiene una fracción menor del tiempo total en las mallas
   grandes — coherente con el término `Θ(√p/n)` del modelo.

---

## 5. Bibliografía y uso de IA (rúbrica punto 3)

Ya está cubierto y referenciado correctamente:

- **`README.md`** cita 2 fuentes (NMSU HPC, LAMMPS docs) para la metodología de medición
  de *speedup*/eficiencia y la distinción *strong* vs *weak scaling*.
- **`docs/pram-y-complejidad.md`** cita **10 fuentes** agrupadas por rol (modelo PRAM /
  work-depth / Brent; isoeficiencia y descomposición de dominio; estabilidad CFL y
  convergencia; documentación MPI), cada una con una frase explicando **para qué se
  usó**.
- **Declaración explícita de uso de IA**: el documento PRAM incluye la nota
  > *"Este documento se redactó con asistencia de IA (Claude) para estructurar el
  > análisis trabajo-profundidad y localizar las referencias canónicas; las fórmulas se
  > derivaron del esquema numérico real del código y se contrastaron con las fuentes
  > citadas arriba."*

  Esto cumple directamente con el criterio "Logrado" de la rúbrica (*"Justifica el uso
  de fuentes externas (publicaciones, charlas, IA)"*).

✅ **Estado: Logrado.** Solo falta trasladar/resumir esta sección al informe consolidado.

---

## 6. Estructura actual del repositorio

```
ahpc-heat-equation/
├── README.md                      # overview, uso, resultados, referencias, equipo
├── assignment.pdf                 # enunciado del proyecto (igual al de Canvas)
├── Makefile                       # make → bin/heat-big, bin/heat-mpi-big
├── src/
│   ├── heat-big.c                 # solver serial
│   └── heat-mpi-big.c             # solver MPI (dominio 2D + halos)
├── scripts/
│   ├── run_local.sh               # smoke test local
│   ├── benchmark.slurm            # barrido p×n en Khipu (SLURM)
│   └── collect_results.sh         # parsea salida del solver -> CSV
├── analysis/
│   ├── plot_performance.py        # CSV -> tiempo/speedup/eficiencia/iteraciones
│   └── requirements.txt
├── results/
│   ├── EXPERIMENTS.md             # bitácora de experimentos
│   ├── exp01-1node-16core/        # CSV + 4 figuras
│   └── exp02-2node-8core/         # CSV + 4 figuras
└── docs/
    ├── heat-equation.md           # guía didáctica (física -> código)
    ├── pram-y-complejidad.md      # objetivo (a): PRAM + complejidad
    ├── generate_figures.py        # genera las figuras de heat-equation.md
    └── img/                       # figuras generadas
```

---

## 7. Pendientes para la **Entrega Parcial** (antes del domingo 14/06/2026)

Ordenados por prioridad:

1. ✅ **Completado — % de participación definido.** Los 4 integrantes (Hugo Angeles,
   Christian Cajusol, Jhomar Yurivilca, Francisco Meza) quedaron registrados al **100%**
   en la Tabla "Participación del equipo" de `Informe_Parcial.tex`.

2. ✅ **Completado — informe parcial armado como documento único.**
   `Informe_Parcial.tex` (→ `Informe_Parcial.pdf`, 20 páginas) cubre exactamente las
   secciones que pide 7.1:
   - Nombre del proyecto + integrantes
   - % de participación
   - Introducción
   - Método = PRAM + métricas (basado en `docs/pram-y-complejidad.md`)
   - Resultados preliminares (exp01/exp02)
   - Bibliografía (10+ referencias) + nota de uso de IA

3. ✅ **Documentado como limitación conocida — `itmax=20000` vs. convergencia.** En
   **ambos** experimentos, `iterations = 20001` para **todas** las combinaciones de `n`
   y `p` — el solver llegó al tope `itmax` sin cumplir `dphimax < eps` (curva "measured"
   plana en `iterations.png` en vez de seguir `Θ(n²)`). El informe parcial ya incluye una
   subsección dedicada ("Validación de T=Θ(n²): limitación conocida") que explica esto y
   lo deja como trabajo pendiente para la entrega final:
   - Aumentar `itmax` (o quitar el tope) para `n=128/256` y verificar que sí convergen
     cerca de las ~14 300 · (n/80)² iteraciones predichas.
   - El barrido actual mide **tiempo por un número fijo de iteraciones** (proxy válido
     de costo por paso, `Θ(n²/p)`); la validación de `T=Θ(n²)`
     (iteraciones-hasta-converger) queda como experimento *aparte* para la entrega final.

4. **🟢 (Opcional, mejora la nota del punto 1 de la rúbrica)** Superponer en
   `speedup.png`/`efficiency.png` la curva teórica con comunicación
   `Ep = 1/(1 + b·√p/n)` (sección 9 de `pram-y-complejidad.md`) ajustando `b` por
   mínimos cuadrados, tal como el propio documento PRAM propone. Esto cerraría el ciclo
   "PRAM → predicción → experimento → validación" de forma muy convincente.

5. **🔴 Pendiente — subir el PDF a Canvas.** `Informe_Parcial.pdf` ya está generado
   (20 páginas); falta subirlo + link al repo a Canvas (Proyecto HPC), como exige la
   sección 7 del enunciado.

---

## 8. Pendientes para la **Entrega Final** (antes del viernes 17/07/2026)

No bloquean la entrega parcial, pero conviene tenerlos en el radar:

- **Más tamaños de problema**: `EXPERIMENTS.md` ya deja anotado `2048, 4096` como
  "planned/next experiments" para explorar el régimen *memory-bandwidth-bound* y
  *weak scaling*.
- **Sección "Resultados y análisis"** (punto 2 de la rúbrica final, 4 pts): describir
  hallazgos (ej. el resultado contraintuitivo de 2 nodos > 1 nodo) y **plantear
  mejoras** a la solución (ej. `stride` adaptativo, comunicación bloqueante vs no
  bloqueante, *overlap* de cómputo/comunicación).
- **Cerrar el punto 3 de §7 anterior**: experimento dedicado a iteraciones-hasta-
  converger para validar `T=Θ(n²)` con varios `n`.
- **Ajuste de constantes del modelo** (`a, b, c` de
  `Tpar(n,p) ≈ a·n⁴/p + b·n³/√p + c`) por mínimos cuadrados sobre el CSV — ya descrito
  en `pram-y-complejidad.md` §9, falta implementarlo en `analysis/plot_performance.py`.
- **Informe final** con la estructura de la sección 7.3 (Resultados y análisis en vez
  de solo Método) + adjuntar/linkear el código (ya está en GitHub, solo falta el link
  explícito en el PDF final).

---

## 9. Cómo correr/reproducir lo que ya existe

```bash
# Local (smoke test)
make                      # compila bin/heat-big y bin/heat-mpi-big
scripts/run_local.sh 4    # corre el solver MPI con 4 procesos
python analysis/plot_performance.py   # genera gráficas desde results/benchmark.csv

# En Khipu (UTEC)
ssh <usuario>@khipu.utec.edu.pe
git clone git@github.com:hugoangeles0810/ahpc-heat-equation.git
cd ahpc-heat-equation
module load gnu12/12.4.0 openmpi4/4.1.6
make mpi
sbatch scripts/benchmark.slurm
squeue --me
```

---

## 10. Resumen ejecutivo para la reunión de equipo

| Pregunta | Respuesta corta |
|---|---|
| ¿El código funciona? | ✅ Sí, serial y MPI, parametrizados, probados en Khipu |
| ¿Está el PRAM (objetivo a)? | ✅ Sí, muy completo — punto fuerte del proyecto |
| ¿Hay mediciones reales (objetivo b)? | ✅ Sí, 2 experimentos en Khipu, barrido p×n completo |
| ¿Hay gráficas/análisis (objetivo c)? | ✅ Sí, 4 gráficas por experimento (tiempo, speedup, eficiencia, iteraciones) |
| ¿Está validada la teoría `T=Θ(n²)`? | ⚠️ No todavía — `itmax=20000` corta antes de converger en todas las corridas; documentado como limitación conocida en el informe |
| ¿Está el informe parcial (PDF, formato 7.1)? | ✅ Sí — `Informe_Parcial.pdf` (20 páginas), consolidado |
| ¿Está definido el % de participación? | ✅ Sí — 100% para los 4 integrantes |
| ¿Bibliografía y uso de IA documentados? | ✅ Sí, con justificación explícita |

**En síntesis:** la entrega parcial está **completa y lista para enviar** — código,
PRAM, experimentos, gráficas, % de participación e informe consolidado
(`Informe_Parcial.pdf`, 20 páginas). Solo queda **subir el PDF a Canvas**. La validación
experimental de `T=Θ(n²)` queda documentada como limitación conocida y como trabajo
pendiente para la entrega final (§8).
