package org.encog.workbench.tabs.query.ocr;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTable;

import org.encog.EncogError;
import org.encog.ml.BasicML;
import org.encog.ml.MLClassification;
import org.encog.ml.MLOutput;
import org.encog.ml.MLRegression;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.dialogs.error.ErrorDialog;
import org.encog.workbench.frames.document.tree.ProjectEGFile;
import org.encog.workbench.models.NetworkQueryModel;
import org.encog.workbench.tabs.EncogCommonTab;

public class OCRQueryTab extends EncogCommonTab implements ActionListener {

	private BasicML method;
	private OCRGridPanel panel;
	private JPanel buttonPanel;
	private JButton buttonQuery;
	private JButton buttonDownsample;
	private JButton buttonClear;
	private DrawingEntry entry;
	private JTable outputTable;
	private boolean classification;

	public OCRQueryTab(ProjectEGFile method) {
		super(method);

		this.classification = method instanceof MLClassification;

		this.method = (BasicML) method.getObject();

		this.setLayout(new BorderLayout());
		JPanel body = new JPanel();
		body.setLayout(new GridLayout(1, 2));
		this.add(body, BorderLayout.CENTER);

		this.buttonQuery = new JButton("Query");
		this.buttonDownsample = new JButton("Downsample");
		this.buttonClear = new JButton("Clear");
		this.buttonPanel = new JPanel();
		this.buttonPanel.add(this.buttonQuery);
		this.buttonPanel.add(this.buttonDownsample);
		this.buttonPanel.add(this.buttonClear);
		this.add(this.buttonPanel, BorderLayout.NORTH);

		this.buttonQuery.addActionListener(this);
		this.buttonDownsample.addActionListener(this);
		this.buttonClear.addActionListener(this);

		this.panel = new OCRGridPanel(this.method);
		this.entry = new DrawingEntry();

		JPanel panelLeft = new JPanel();
		JPanel panelRight = new JPanel();
		panelLeft.setLayout(new GridLayout(2, 1));
		panelLeft.add(this.panel);
		panelLeft.add(this.entry);
		
		panelLeft.setBorder(BorderFactory.createLineBorder(Color.BLACK));
		panelRight.setBorder(BorderFactory.createLineBorder(Color.BLACK));

		body.add(panelLeft);
		body.add(panelRight);

		panelRight.setLayout(new BorderLayout());

		if (classification) {
			int outputCount = 1;

			panelRight.add(this.outputTable = new JTable(new NetworkQueryModel(
					outputCount, 2)), BorderLayout.CENTER);
			this.outputTable.setEnabled(false);

			this.outputTable.setValueAt("Classification Output:", 0, 0);
			this.outputTable.setValueAt("0.0", 0, 1);

		} else {
			int outputCount = ((MLOutput) method).getOutputCount();

			panelRight.add(this.outputTable = new JTable(new NetworkQueryModel(
					outputCount, 2)), BorderLayout.CENTER);
			this.outputTable.setEnabled(false);

			for (int i = 1; i <= outputCount; i++) {
				this.outputTable.setValueAt("Output " + i + ":", i - 1, 0);
				this.outputTable.setValueAt("0.0", i - 1, 1);
			}
		}
	}

	public void performDownsample() {
		boolean[] data = this.entry.downSample(this.panel.getGridWidth(),
				this.panel.getGridHeight());
		this.panel.setGrid(data);
	}

	public void performQuery() {
		try {
			int outputCount = ((MLOutput) method).getOutputCount();
			boolean[] grid = this.panel.getGrid();
			NeuralData input = new BasicNeuralData(grid.length);
			for (int i = 0; i < grid.length; i++) {
				input.setData(i, grid[i] ? 1 : -1);
			}

			if (classification) {
				int output = ((MLClassification) this.method).classify(input);
				this.outputTable.setValueAt(output, 0, 1);
			} else {
				NeuralData output = ((MLRegression) this.method).compute(input);
				for (int i = 0; i < outputCount; i++) {
					this.outputTable.setValueAt(output.getData(i), i, 1);
				}
			}
		} catch (EncogError ex) {
			EncogWorkBench.displayError("Query Error", ex.getMessage());
		} catch (Throwable t) {
			ErrorDialog.handleError(t, method, null);
		}
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == this.buttonClear) {
			this.panel.clear();
			this.entry.clear();
		}
		if (e.getSource() == this.buttonDownsample) {
			performDownsample();
		}
		if (e.getSource() == this.buttonQuery) {
			performQuery();
		}

	}

}
