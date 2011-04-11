package org.encog.workbench.tabs.visualize.scatter;

import java.awt.Color;
import java.awt.GridLayout;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.encog.app.analyst.EncogAnalyst;
import org.encog.workbench.tabs.EncogCommonTab;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.data.xy.XYDataset;

public class ScatterPlotTab extends EncogCommonTab {

	private EncogAnalyst analyst;
	private ScatterFile file;
	
	public ScatterPlotTab(EncogAnalyst analyst, String className, List<String> axisList) {
		super(null);
		this.analyst = analyst;
		this.file = new ScatterFile(this.analyst,className,axisList);
		
		if( axisList.size()<=2 ) {
			this.add(createPanel(0,1));
			return;
		} else {
			int count = axisList.size();
			this.setLayout(new GridLayout(count,count));
			
			for(int col=0;col<count;col++) {
				for(int row=0;row<count;row++) {
					if( col==row ) {
						this.add(new ScatterLabelPane(axisList.get(row)));
					} else {
						this.add(createPanel(row,col));			
					}											
				}				
			}
		}
	}
	
	private JPanel createPanel(int xIndex, int yIndex) {
		
		 XYDataset dataset = new ScatterXY(file,xIndex,yIndex);
	        JFreeChart chart = ChartFactory.createScatterPlot(null,
	            null, null, dataset, PlotOrientation.VERTICAL, false, true, false);

	        XYPlot plot = (XYPlot) chart.getPlot();

	        XYDotRenderer renderer = new XYDotRenderer();
	        renderer.setDotWidth(4);
	        renderer.setDotHeight(4);
	        plot.setRenderer(renderer);
	        plot.setDomainCrosshairVisible(true);
	        plot.setRangeCrosshairVisible(true);

	        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
	        domainAxis.setAutoRangeIncludesZero(false);
	        plot.getRangeAxis().setInverted(false);
	        
	        ChartPanel result = new ChartPanel(chart); 	
	        result.setBorder(BorderFactory.createLineBorder(Color.black));
	        return result;
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Scatter Plot";
	}

}
