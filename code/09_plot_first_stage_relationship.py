plt.figure(figsize=(8, 6))

plt.scatter(
    df["pred_tariff_shock"],
    df["china_share"],
    alpha=0.15,
    s=10
)

df_sorted = df.sort_values("pred_tariff_shock")
X_sorted = sm.add_constant(df_sorted["pred_tariff_shock"])
pred_vals = model.predict(X_sorted)

plt.plot(
    df_sorted["pred_tariff_shock"],
    pred_vals,
    linewidth=2.5
)

plt.grid(alpha=0.2)

plt.xlim(0, df["pred_tariff_shock"].quantile(0.99))

plt.xlabel("Predicted Tariff Exposure", fontsize=12)
plt.ylabel("Share of Product Imports Sourced from China", fontsize=12)
plt.title(
    "First-Stage Relationship Between Product-Level Tariff Exposure\n"
    "and China Import Share, 2015–2020",
    fontsize=13
)

plt.figtext(
    0.5,
    0.01,
    "Notes: Observations are at the product-year level. Predicted tariff exposure is constructed using pre-2018 import shares and Section 301 tariff changes.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(fig_path, dpi=300)
plt.show()

print("\nSaved figure to:", fig_path)