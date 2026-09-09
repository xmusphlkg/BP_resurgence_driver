#!/usr/bin/env Rscript

source(file.path("script", "figures", "_common.R"))

lattice <- read_csv_checked(
  fd_path("model2c_external_validation_claim_lattice.csv"),
  c(
    "iso3",
    "country",
    "validation_domain",
    "evidence_available",
    "evidence_type",
    "claim_strength",
    "claim_strength_score",
    "model2c_output_version"
  )
) %>%
  require_model2c_current_version("Model 2C external validation audit") %>%
  mutate(
    evidence_available = as.logical(evidence_available),
    country = short_country(country),
    claim_class = case_when(
      !evidence_available ~ "not_available",
      str_detect(claim_strength, "real_transmission") ~ "real_transmission",
      str_detect(claim_strength, "reporting") ~ "reporting",
      str_detect(claim_strength, "context") ~ "context",
      str_detect(claim_strength, "contact") ~ "contact",
      str_detect(claim_strength, "age_pattern") ~ "age_pattern",
      TRUE ~ "immune_landscape"
    ),
    available_label = if_else(evidence_available, "1", "0")
  )

domain_order <- c(
  "serology",
  "direct_testing_denominator",
  "official_severe_outcomes",
  "pathogen_amr_timeline",
  "contact_validation",
  "age_pattern_evidence",
  "reporting_practice_case_definition"
)

domain_labels <- c(
  serology = "Serology",
  direct_testing_denominator = "Testing\ndenominator",
  official_severe_outcomes = "Severe\noutcomes",
  pathogen_amr_timeline = "Pathogen/\nAMR",
  contact_validation = "Contact\nvalidation",
  age_pattern_evidence = "Age-pattern\nevidence",
  reporting_practice_case_definition = "Reporting\npractice"
)

country_order <- lattice %>%
  group_by(iso3, country) %>%
  summarise(score = sum(evidence_available, na.rm = TRUE), .groups = "drop") %>%
  arrange(score, iso3) %>%
  pull(iso3)

lattice <- lattice %>%
  mutate(
    iso3 = factor(iso3, levels = country_order),
    validation_domain = factor(validation_domain, levels = domain_order, labels = unname(domain_labels)),
    claim_class = factor(
      claim_class,
      levels = c(
        "real_transmission",
        "reporting",
        "context",
        "contact",
        "age_pattern",
        "immune_landscape",
        "not_available"
      )
    )
  )

claim_palette <- c(
  real_transmission = clr("green"),
  reporting = clr("dark_grey"),
  context = clr("red"),
  contact = clr("blue"),
  age_pattern = clr("teal"),
  immune_landscape = clr("gold"),
  not_available = pal["light_grey"]
)

heatmap_panel <- ggplot(lattice, aes(x = validation_domain, y = iso3, fill = claim_class)) +
  geom_tile(colour = "white", linewidth = 0.22) +
  geom_text(aes(label = available_label), size = figure_text$annotation_large, colour = clr("black")) +
  scale_fill_manual(
    values = claim_palette,
    breaks = c("real_transmission", "reporting", "context", "contact", "age_pattern", "immune_landscape", "not_available"),
    labels = c(
      "Real transmission",
      "Reporting visibility",
      "Context audit",
      "Contact proxy",
      "Age pattern",
      "Immune landscape",
      "Not available"
    ),
    name = "Evidence role",
    drop = FALSE
  ) +
  labs(x = "External evidence domain", y = "Country") +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
  theme_nature() +
  theme(
    axis.text.x = element_text(angle = 34, hjust = 1, vjust = 1, size = figure_text$axis_text),
    axis.text.y = element_text(size = figure_text$axis_text),
    panel.grid = element_blank()
  )

score_panel <- lattice %>%
  group_by(iso3) %>%
  summarise(available_domains = sum(evidence_available, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    iso3 = factor(iso3, levels = levels(lattice$iso3)),
    count_label = paste0(available_domains, "/7")
  ) %>%
  ggplot(aes(y = iso3, x = available_domains)) +
  geom_col(width = 0.70, fill = clr("green"), colour = "white", linewidth = 0.18) +
  geom_text(
    aes(x = available_domains + 0.15, label = count_label),
    size = figure_text$annotation_large,
    hjust = 0,
    colour = clr("black")
  ) +
  scale_x_continuous(limits = c(0, 7.8), breaks = 0:7, expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Available domains", y = NULL) +
  theme_nature() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

figure <- panel_tags(
  (heatmap_panel | score_panel) +
    plot_layout(widths = c(1.55, 0.55), guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = figure_text$legend_text),
      legend.title = element_text(size = figure_text$legend_title)
    )
)

save_nature(figure, "supplement_model2c_external_validation_lattice.png", nature_full_width_mm, 130)
