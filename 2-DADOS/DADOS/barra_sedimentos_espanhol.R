# ─────────────────────────────────────────────
# 1. Paquetes
# ─────────────────────────────────────────────
paquetes <- c("readxl", "dplyr", "stringr", "ggplot2", "tidyr", "ggpattern")
invisible(lapply(paquetes, require, character.only = TRUE))

rm(list = ls()); gc()

# ─────────────────────────────────────────────
# 2. Importar y preparar los datos
# ─────────────────────────────────────────────
datos_raw <- read_excel("BD.xlsx", sheet = "GRAFICO")

datos <- datos_raw |>
  mutate(
    across(c(SEDIMENT, RAINFALL, FRACIONADO),
           ~ as.numeric(str_replace_all(., ",", "."))),
    RAINFALL   = replace_na(RAINFALL, 0),
    FRACIONADO = replace_na(FRACIONADO, 0),
    FECHA  = as.Date(paste(YEAR, MONTH, "01", sep = "-")),
    ETIQUETA = format(FECHA, "%b\n%Y") |> str_to_title(),
    AREA = factor(
      AREA,
      levels = c("SUP", "MED", "INF"),
      labels = c("Porción superior del talud",
                 "Porción media del talud",
                 "Porción inferior del talud")
    )
  )

# ─────────────────────────────────────────────
# 3. Línea de precipitación
# ─────────────────────────────────────────────
linea_lluvia <- datos |>
  group_by(FECHA, ETIQUETA) |>
  summarise(RAINFALL = max(RAINFALL), .groups = "drop") |>
  mutate(Serie = "Precipitación")

# ─────────────────────────────────────────────
# 4. Factor de escala (para lluvia)
# ─────────────────────────────────────────────
max_lluvia <- max(linea_lluvia$RAINFALL, na.rm = TRUE)
max_frac   <- max(datos$FRACIONADO, na.rm = TRUE)

factor_escala <- if (is.finite(max_lluvia) && is.finite(max_frac) && max_frac > 0) {
  max_lluvia / max_frac
} else {
  1
}

linea_lluvia <- mutate(linea_lluvia, LLUVIA_ESC = RAINFALL / factor_escala)

# ─────────────────────────────────────────────
# 5. Gráfico final
# ─────────────────────────────────────────────
ggplot() +
  ## Barras con textura (generación mensual de sedimentos)
  geom_col_pattern(
    data = datos,
    aes(x = FECHA, y = FRACIONADO,
        pattern = AREA, pattern_fill = AREA),
    position = position_dodge(width = 20),
    width = 15, colour = "black",
    fill = NA,
    pattern_density = 0.5,
    pattern_spacing = 0.03,
    pattern_key_scale_factor = 0.7
  ) +
  
  ## Línea de precipitación (línea discontinua roja)
  geom_line(
    data = linea_lluvia,
    aes(x = FECHA, y = LLUVIA_ESC, color = Serie, linetype = Serie),
    linewidth = 1.1
  ) +
  geom_point(
    data = linea_lluvia,
    aes(x = FECHA, y = LLUVIA_ESC, color = Serie, shape = Serie, fill = Serie),
    size = 2.5, stroke = 0.8
  ) +
  
  ## Escalas de los ejes
  scale_y_continuous(
    name = "Generación de sedimentos (m)",
    sec.axis = sec_axis(~ . * factor_escala, name = "Precipitación promedio (mm)")
  ) +
  scale_x_date(
    breaks = datos$FECHA,
    labels = datos$ETIQUETA,
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  
  ## Patrones de textura por posición
  scale_pattern_manual(
    values = c("Porción superior del talud" = "stripe",
               "Porción media del talud"    = "polygon_tiling",
               "Porción inferior del talud" = "pch"),
    name = "Posición del sedimento"
  ) +
  scale_pattern_fill_manual(
    values = c("Porción superior del talud" = "black",
               "Porción media del talud"    = "grey30",
               "Porción inferior del talud" = "grey10"),
    guide = "none"
  ) +
  
  ## Estilo para la línea de precipitación
  scale_color_manual(
    values = c("Precipitación" = "red"),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c("Precipitación" = "dashed"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Precipitación" = 21),
    name = NULL
  ) +
  scale_fill_manual(
    values = c("Precipitación" = "white"),
    guide = "none"
  ) +
  
  ## Tema del gráfico
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
    title = "Generación mensual de sedimentos y precipitación (jul/2023 – may/2025)"
  )
