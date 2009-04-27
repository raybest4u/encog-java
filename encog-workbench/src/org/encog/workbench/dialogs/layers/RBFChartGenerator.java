package org.encog.workbench.dialogs.layers;

import java.awt.BasicStroke;
import java.awt.Color;

import org.encog.neural.networks.layers.RadialBasisFunctionLayer;
import org.encog.workbench.dialogs.common.ChartGenerator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYPointerAnnotation;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.function.Function2D;
import org.jfree.data.function.NormalDistributionFunction2D;
import org.jfree.data.general.DatasetUtilities;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.TextAnchor;

public class RBFChartGenerator implements ChartGenerator {

	private RadialBasisFunctionLayer layer;
	
	public RBFChartGenerator(RadialBasisFunctionLayer layer)
	{
		this.layer = layer;
	}
	
	public JFreeChart createChart(XYDataset dataset) {
		JFreeChart chart = ChartFactory.createXYLineChart(
	            null,
	            "X",
	            "Y",
	            dataset,
	            PlotOrientation.VERTICAL,
	            false,
	            true,
	            false
	        );
	        XYPlot plot = (XYPlot) chart.getPlot();
	        plot.setDomainZeroBaselineVisible(true);
	        plot.setRangeZeroBaselineVisible(true);
	        plot.setDomainPannable(true);
	        plot.setRangePannable(true);
	        ValueAxis xAxis = plot.getDomainAxis();
	        xAxis.setLowerMargin(0.0);
	        xAxis.setUpperMargin(0.0);
	        XYLineAndShapeRenderer r = (XYLineAndShapeRenderer) plot.getRenderer();
	        
	        for(int i=0;i<this.layer.getRadialBasisFunction().length;i++)
	        {
	        	r.setDrawSeriesLineAsPath(true);
		        r.setSeriesStroke(i, new BasicStroke(1.5f));	
	        }
	        
	        
	        


	        return chart;
	}

	public XYDataset createDataset() {
        XYSeriesCollection dataset = new XYSeriesCollection();

        for(int i=0;i<this.layer.getRadialBasisFunction().length;i++)
        {
        	Function2D n1 = new RadialBasisFunction2D(this.layer.getRadialBasisFunction()[i]);
            XYSeries s1 = DatasetUtilities.sampleFunction2DToSeries(n1, -5.1, 5.1,
                    121, "N1");
            dataset.addSeries(s1);
        }
        
        


        return dataset;
	}

}
