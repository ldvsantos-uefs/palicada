# ─────────────────────────────────────────────
# 1. Packages
# ─────────────────────────────────────────────
packages <- c("readxl", "dplyr", "stringr", "ggplot2", "tidyr", "ggpattern")
invisible(lapply(packages, require, character.only = TRUE))

rm(list = ls()); gc()

# ─────────────────────────────────────────────
# 2. Import and prepare data
# ─────────────────────────────────────────────
raw_data <- read_excel("BD.xlsx", sheet = "GRAFICO")

data <- raw_data |>
  mutate(
    across(c(SEDIMENT, RAINFALL, FRACIONADO),
           ~ as.numeric(str_replace_all(., ",", "."))),
    RAINFALL    = replace_na(RAINFALL,   0),
    FRACIONADO  = replace_na(FRACIONADO, 0),
    DATE  = as.Date(paste(YEAR, MONTH, "01", sep = "-")),
    LABEL = format(DATE, "%b\n%Y") |> str_to_title(),
    AREA = factor(
      AREA,
      levels = c("SUP", "MED", "INF"),
      labels = c("Upper Slope Portion",
                 "Middle Slope Portion",
                 "Lower Slope Portion")
    )
  )

# ─────────────────────────────────────────────
# 3. Rainfall and Sediment Series
# ─────────────────────────────────────────────
rain_line <- data |>
  group_by(DATE, LABEL) |>
  summarise(RAINFALL = max(RAINFALL), .groups = "drop") |>
  mutate(Series = "Rainfall")

frac_line <- data |>
  mutate(Series = "Sediment") |>
  rename(FRAC = FRACIONADO) |>
  select(DATE, LABEL, FRAC, Series)

# ─────────────────────────────────────────────
# 4. Scaling factor (for rainfall)
# ─────────────────────────────────────────────
max_rain <- max(rain_line$RAINFALL, na.rm = TRUE)
max_sed  <- max(data$SEDIMENT,     na.rm = TRUE)

if (is.finite(max_rain) && is.finite(max_sed) && max_sed > 0) {
  scale_factor <- max_rain / max_sed
} else {
  scale_factor <- 1
}

rain_line <- mutate(rain_line, RAINFALL_SCALED = RAINFALL / scale_factor)

# ─────────────────────────────────────────────
# 5. Final Plot
# ─────────────────────────────────────────────
ggplot() +
  ## Patterned bars (accumulated sediment)
  geom_col_pattern(
    data = data,
    aes(x = DATE, y = SEDIMENT,
        pattern = AREA, pattern_fill = AREA),
    position = position_dodge(width = 20),
    width = 15, colour = "black",
    fill = NA,
    pattern_density = 0.5,
    pattern_spacing = 0.03,
    pattern_key_scale_factor = 0.7
  ) +
  
  ## Sediment line (solid blue) + filled points
  geom_line(
    data = frac_line,
    aes(x = DATE, y = FRAC, color = Series, linetype = Series),
    linewidth = 1.1
  ) +
  geom_point(
    data = frac_line,
    aes(x = DATE, y = FRAC, color = Series, shape = Series),
    size = 2.5
  ) +
  
  ## Rainfall line (dashed red) + hollow points
  geom_line(
    data = rain_line,
    aes(x = DATE, y = RAINFALL_SCALED, color = Series, linetype = Series),
    linewidth = 1.1
  ) +
  geom_point(
    data = rain_line,
    aes(x = DATE, y = RAINFALL_SCALED, color = Series, shape = Series, fill = Series),
    size = 2.5, stroke = 0.8
  ) +
  
  ## Axis scales
  scale_y_continuous(
    name = "Sediment Input (m)",
    sec.axis = sec_axis(~ . * scale_factor, name = "Average Rainfall (mm)")
  ) +
  scale_x_date(
    breaks = data$DATE,
    labels = data$LABEL,
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  
  ## Patterns for bars
  scale_pattern_manual(
    values = c("Upper Slope Portion"  = "stripe",
               "Middle Slope Portion" = "polygon_tiling",
               "Lower Slope Portion"  = "pch"),
    name = "Sediment Position"
  ) +
  scale_pattern_fill_manual(
    values = c("Upper Slope Portion"  = "black",
               "Middle Slope Portion" = "grey30",
               "Lower Slope Portion"  = "grey10"),
    guide = "none"
  ) +
  
  ## Color and symbol settings for rainfall/sediment lines
  scale_color_manual(
    values = c("Rainfall" = "red",
               "Sediment" = "blue"),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c("Rainfall" = "dashed",
               "Sediment" = "solid"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Rainfall" = 21,
               "Sediment" = 16),
    name = NULL
  ) +
  scale_fill_manual(
    values = c("Rainfall" = "white",
               "Sediment" = "blue"),
    guide = "none"
  ) +
  
  ## Theme
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
    title = "Monthly Sediment Input, Fractional Sediment and Rainfall (Jul/2023 – May/2025)"
  )
