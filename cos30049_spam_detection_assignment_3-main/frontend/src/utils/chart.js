// utils/charts.js
import { Chart as ChartJS, defaults } from "chart.js";

// cấu hình mặc định cho toàn bộ chart
export function setupChartDefaults() {
    ChartJS.register();
    defaults.color = "#e5e7eb";
    defaults.font.family = "ui-sans-serif, system-ui";
    defaults.plugins.legend.labels.boxWidth = 12;
    defaults.plugins.legend.labels.color = "#94a3b8";
    defaults.plugins.tooltip.backgroundColor = "rgba(15,23,42,0.9)";
    defaults.plugins.tooltip.borderColor = "#38bdf8";
    defaults.plugins.tooltip.borderWidth = 1;
    defaults.plugins.tooltip.titleColor = "#38bdf8";
    defaults.plugins.tooltip.bodyColor = "#e2e8f0";
}
