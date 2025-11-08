import { Chart as ChartJS, defaults } from "chart.js";

export function setupChartDefaults() {
    ChartJS.register();
    defaults.color = "#FFFFFF";  
    defaults.font.family = "ui-sans-serif, system-ui";
    defaults.plugins.legend.labels.boxWidth = 12;
    defaults.plugins.legend.labels.color = "#FFFFFF";  
    defaults.plugins.tooltip.backgroundColor = "rgba(0,0,0,0.8)";
    defaults.plugins.tooltip.borderColor = "#FFFFFF";
    defaults.plugins.tooltip.borderWidth = 1;
    defaults.plugins.tooltip.titleColor = "#FFFFFF";
    defaults.plugins.tooltip.bodyColor = "#FFFFFF";
}
