package org.encog.workbench.tabs;

import java.awt.BorderLayout;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;

import javax.swing.JButton;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JToolBar;

import org.encog.EncogError;
import org.encog.mathutil.libsvm.svm_model;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.data.buffer.BufferedNeuralDataSet;
import org.encog.persist.EncogPersistedObject;
import org.encog.workbench.models.BufferedDataSetTableModel;
import org.encog.workbench.models.TrainingSetTableModel;
import org.encog.workbench.process.ImportExport;
import org.encog.workbench.util.EncogFonts;
import org.encog.workbench.util.ExcelAdapter;

public class ExternalLinkTab extends EncogCommonTab implements ActionListener {

	private BufferedNeuralDataSet object;
	private BufferedDataSetTableModel model;
	private JToolBar toolbar;
	private JTable table;

	private JButton addInputColumn;
	private JButton delColumn;
	private JButton addIdealColumn;
	private JButton addRow;
	private JButton delRow;
	private JButton export;
	private RandomAccessFile raf;
	private FileChannel fc;
	
	public ExternalLinkTab(EncogPersistedObject encogObject) {
		super(encogObject);
		this.object = (BufferedNeuralDataSet)encogObject;
		setLayout(new BorderLayout());
		this.toolbar = new JToolBar();
		this.toolbar.setFloatable(false);
		this.toolbar.add(this.addInputColumn = new JButton("Add Input Column"));
		this.toolbar.add(this.delColumn = new JButton("Delete Column"));
		this.toolbar.add(this.addIdealColumn = new JButton("Add Ideal Column"));
		this.toolbar.add(this.addRow = new JButton("Add Row"));
		this.toolbar.add(this.delRow = new JButton("Delete Row"));
		this.toolbar.add(this.export = new JButton("Export"));
		this.addInputColumn.addActionListener(this);
		this.delColumn.addActionListener(this);
		this.addIdealColumn.addActionListener(this);
		this.addRow.addActionListener(this);
		this.delRow.addActionListener(this);
		this.export.addActionListener(this);
		add(this.toolbar, BorderLayout.PAGE_START);
		this.model = new BufferedDataSetTableModel(getData());
		this.table = new JTable(this.model);
		add(new JScrollPane(this.table), BorderLayout.CENTER);
		this.table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		//new ExcelAdapter( this.table );
		this.object.close();
		try {
			this.raf = new RandomAccessFile(this.object.getFile(),"rw");
			this.fc = this.raf.getChannel();
		} catch (FileNotFoundException e) {
			throw new EncogError(e);
		}
	}
	
	/**
	 * Paint the panel.
	 * @param g The graphics object to use.
	 */
	/*public void paint(final Graphics g) {
		super.paint(g);
		final FontMetrics fm = g.getFontMetrics();
		g.setFont(EncogFonts.getInstance().getTitleFont());
		int y = fm.getHeight();
		g.setFont(EncogFonts.getInstance().getTitleFont());
		g.drawString("External Data Source", 0, y);
		y += g.getFontMetrics().getHeight();
		
		g.setFont(EncogFonts.getInstance().getHeadFont());
		g.drawString("File:", 10, y);
		g.setFont(EncogFonts.getInstance().getBodyFont());
		g.drawString(this.object.getFile().toString(), 150, y);
		y += g.getFontMetrics().getHeight();
		
		g.setFont(EncogFonts.getInstance().getHeadFont());
		g.drawString("Input Count:", 10, y);
		g.setFont(EncogFonts.getInstance().getBodyFont());
		g.drawString(""+this.object.getInputSize(), 150, y);
		y += g.getFontMetrics().getHeight();
		
		g.setFont(EncogFonts.getInstance().getHeadFont());
		g.drawString("Ideal Count:", 10, y);
		g.setFont(EncogFonts.getInstance().getBodyFont());
		g.drawString(""+this.object.getIdealSize(), 150, y);
		y += g.getFontMetrics().getHeight();
		
		g.setFont(EncogFonts.getInstance().getHeadFont());
		g.drawString("Record Count:", 10, y);
		g.setFont(EncogFonts.getInstance().getBodyFont());
		g.drawString(""+this.object.getRecordCount(), 150, y);
		y += g.getFontMetrics().getHeight();
		
	}*/
	
	public void actionPerformed(final ActionEvent action) {
		final int row = this.table.getSelectedRow();
		final int col = this.table.getSelectedColumn();

		if (action.getSource() == this.addInputColumn) {
			this.model.addInputColumn();
		} else if (action.getSource() == this.delColumn) {
			if (col == -1) {
				JOptionPane.showMessageDialog(this,
						"Please move to the column you wish to delete.",
						"Error", JOptionPane.ERROR_MESSAGE);
			} else if (col < getData().getInputSize()
					&& getData().getInputSize() <= 1) {
				JOptionPane.showMessageDialog(this,
						"There must be at least one input column.", "Error",
						JOptionPane.ERROR_MESSAGE);
			} else {
				this.model.delColumn(col);
			}
		} else if (action.getSource() == this.addIdealColumn) {
			this.model.addIdealColumn();
		} else if (action.getSource() == this.addRow) {
			this.model.addRow(row);
		} else if (action.getSource() == this.delRow) {
			if (row == -1) {
				JOptionPane.showMessageDialog(this,
						"Please move to the row you wish to delete.", "Error",
						JOptionPane.ERROR_MESSAGE);
			} else {
				this.model.delRow(row);
			}
		} else if(action.getSource()==this.export)
		{
			ImportExport.performExport(this.getEncogObject());
		}

	}
	
	public void dispose()
	{
		super.dispose();
		try {
			this.fc.close();
			this.raf.close();
		} catch (IOException e) {
			throw new EncogError(e);
		}
		this.object.open();
	}

	public BufferedNeuralDataSet getData() {
		return (BufferedNeuralDataSet) getEncogObject();
	}

}
