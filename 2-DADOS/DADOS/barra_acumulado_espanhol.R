# ─────────────────────────────────────────────
# 1. Paquetes
# ─────────────────────────────────────────────
pacotes <- c("readxl", "dplyr", "stringr", "ggplot2", "tidyr", "ggpattern")
invisible(lapply(pacotes, require, character.only = TRUE))

rm(list = ls()); gc()

# ─────────────────────────────────────────────
# 2. Importar y preparar datos
# ─────────────────────────────────────────────
datos_raw <- read_excel("BD.xlsx", sheet = "GRAFICO")

datos <- datos_raw |>
  mutate(
    across(c(SEDIMENT, RAINFALL, FRACIONADO),
           ~ as.numeric(str_replace_all(., ",", "."))),
    RAINFALL   = replace_na(RAINFALL,   0),
    FRACIONADO = replace_na(FRACIONADO, 0),
    DATA  = as.Date(paste(YEAR, MONTH, "01", sep = "-")),
    LABEL = format(DATA, "%b\n%Y") |> str_to_title(),
    AREA = factor(
      AREA,
      levels = c("SUP", "MED", "INF"),
      labels = c("Porción superior del talud",
                 "Porción intermedia del talud",
                 "Porción inferior del talud")
    )
  )

# ─────────────────────────────────────────────
# 3. Series de precipitación y Sedimentos
# ─────────────────────────────────────────────
rain_line <- datos |>
  group_by(DATA, LABEL) |>
  summarise(RAINFALL = max(RAINFALL), .groups = "drop") |>
  mutate(Serie = "Precipitación")

fracc_line <- datos |>
  mutate(Serie = "Sedimentos") |>
  rename(FRACC = FRACIONADO) |>
  select(DATA, LABEL, FRACC, Serie)

# ─────────────────────────────────────────────
# 4. Factor de escala (para lluvia)
# ─────────────────────────────────────────────
max_rain <- max(rain_line$RAINFALL, na.rm = TRUE)
max_sed  <- max(datos$SEDIMENT,    na.rm = TRUE)

if (is.finite(max_rain) && is.finite(max_sed) && max_sed > 0) {
  escala <- max_rain / max_sed
} else {
  escala <- 1  # valor padrão seguro
}

rain_line <- mutate(rain_line, RAINFALL_ESC = RAINFALL / escala)

# ─────────────────────────────────────────────
# 5. Gráfico final
# ─────────────────────────────────────────────
ggplot() +
  ## Barras con textura (sedimentos acumulados)
  geom_col_pattern(
    data = datos,
    aes(x = DATA, y = SEDIMENT,
        pattern = AREA, pattern_fill = AREA),
    position = position_dodge(width = 20),
    width = 15, colour = "black",
    fill = NA,
    pattern_density = 0.5,
    pattern_spacing = 0.03,
    pattern_key_scale_factor = 0.7
  ) +
  
  ## Línea de Sedimentos (azul continúa) + puntos sólidos
  geom_line(
    data = fracc_line,
    aes(x = DATA, y = FRACC, color = Serie, linetype = Serie),
    linewidth = 1.1
  ) +
  geom_point(
    data = fracc_line,
    aes(x = DATA, y = FRACC, color = Serie, shape = Serie),
    size = 2.5
  ) +
  
  ## Línea discontinua de precipitación (rojo) + puntos huecos
  geom_line(
    data = rain_line,
    aes(x = DATA, y = RAINFALL_ESC, color = Serie, linetype = Serie),
    linewidth = 1.1
  ) +
  geom_point(
    data = rain_line,
    aes(x = DATA, y = RAINFALL_ESC, color = Serie, shape = Serie, fill = Serie),
    size = 2.5, stroke = 0.8
  ) +
  
  ## Escalas de eje
  scale_y_continuous(
    name = "Aporte de sedimentos (m)",
    sec.axis = sec_axis(~ . * escala, name = "Precipitación promedio (mm)")
  ) +
  scale_x_date(
    breaks = datos$DATA,
    labels = datos$LABEL,
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  
  ## Patrones para las barras
  scale_pattern_manual(
    values = c("Porción superior del talud"     = "stripe",
               "Porción intermedia del talud"   = "polygon_tiling",
               "Porción inferior del talud"     = "pch"),
    name   = "Posición del sedimento"
  ) +
  scale_pattern_fill_manual(
    values = c("Porción superior del talud"     = "black",
               "Porción intermedia del talud"   = "grey30",
               "Porción inferior del talud"     = "grey10"),
    guide = "none"
  ) +
  
  ## Escalas para las líneas y puntos
  scale_color_manual(
    values = c("Precipitación"   = "blue",
               "Sedimentos" = "red"),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c("Precipitación"   = "dashed",
               "Sedimentos" = "solid"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Precipitación"   = 21,
               "Sedimentos" = 16),
    name = NULL
  ) +
  scale_fill_manual(
    values = c("Precipitación" = "blue",
               "Sedimentos" = "red"),
    guide = "none"
  ) +
  
  ## Tema
  theme_classic(base_size = 12) +
  theme(
    axis.text.x       = element_text(size = 9, angle = 45, hjust = 1),
    axis.title.x      = element_blank(),
    plot.margin       = margin(20, 20, 40, 20),
    legend.position   = "bottom",
    legend.box        = "vertical",
    legend.title      = element_text(face = "bold"),
    legend.background = element_rect(colour = "black", fill = "white")
  ) +
  labs(
    title = "Aporte mensual de sedimentos, Sedimentos y precipitación (jul/2023 – may/2025)"
  )
