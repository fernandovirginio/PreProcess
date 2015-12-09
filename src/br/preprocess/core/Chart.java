package br.preprocess.core;

import java.awt.Color;
import java.util.ArrayList;
import javax.swing.JPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public final class Chart {
    
    public static final int CHART_LINE = 0;
    public static final int CHART_HISTOGRAM = 1;
    public static final int CHART_SCATTER = 2;

    private JFreeChart jFreeChart;
    private final XYSeriesCollection dataset = new XYSeriesCollection();
    private final ArrayList<XYSeries> xySeries = new ArrayList<>();
    private ChartPanel img;
    private final int height, width;

    public Chart(int width, int height, String t, String x, String y, int type, boolean hiddenDesc) {
        this.height = height;
        this.width = width;
        this.draw(t, x, y, type, hiddenDesc);
    }
    
    public int addXYSerie(String name) {
        XYSeries xy = new XYSeries(name);
        this.xySeries.add(xy);
        this.dataset.addSeries(xy);
        return this.xySeries.size()-1;
    }
    
    public void update(double x, double y, int index) {
        this.xySeries.get(index).add(x, y);
        this.jFreeChart.fireChartChanged();
    }

    public void clear() {
        this.xySeries.stream().forEach((xy) -> {
            xy.clear();
        });
    }

    public JPanel getImage() {
        return this.img;
    }

    private void draw(String t, String x, String y, int type, boolean hiddenDesc) {
        switch (type) {
            case Chart.CHART_LINE:
                this.jFreeChart = ChartFactory.createXYLineChart(t, x, y, this.dataset, PlotOrientation.VERTICAL, true, true, true);
                break;
            case Chart.CHART_HISTOGRAM:
                this.jFreeChart = ChartFactory.createHistogram(t, x, y, this.dataset, PlotOrientation.VERTICAL, true, true, true);
                break;
            case Chart.CHART_SCATTER:
                this.jFreeChart = ChartFactory.createScatterPlot(t, x, y, this.dataset, PlotOrientation.VERTICAL, true, true, true);
                break;
            default:
                return;
        }
        if (hiddenDesc)
            this.jFreeChart.removeLegend();
        final XYPlot plot = (XYPlot) this.jFreeChart.getPlot();
        plot.setBackgroundPaint(new Color(240, 240, 240));
        NumberAxis axis = (NumberAxis) plot.getRangeAxis();
        if (type != Chart.CHART_HISTOGRAM) axis.setAutoRangeIncludesZero(false);
        axis.configure();
        this.img = new ChartPanel(this.jFreeChart);
        this.img.setPreferredSize(new java.awt.Dimension(this.width, this.height - 2));
        this.img.setMinimumSize(new java.awt.Dimension(this.width, this.height - 2));
        this.img.setMaximumSize(new java.awt.Dimension(this.width, this.height - 2));
        this.img.setMouseZoomable(true);
        this.img.updateUI();
    }
}